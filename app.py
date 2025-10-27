# app.py — Lighting Catalog RAG (OCR + Folder Auto-Load + Cache)
# 依賴：pip install -U gradio pdfplumber sentence-transformers scikit-learn numpy pandas pillow pymupdf pytesseract
# 啟動前先開變數:venv\Scripts\activate
import os, io, re, pickle, logging, warnings
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import gradio as gr
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---- 靜音一些不是錯誤的噪音訊息 ----
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="fitz")

# =========================
# 基本設定（可自行調整）
# =========================
EMBED_MODEL_NAME = "BAAI/bge-m3"                 # 多語向量模型
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"     # 交叉編碼 rerank
USE_RERANK = True                                # 初期為 True；若要更快可改 False

DEFAULT_OCR_LANG = "chi_tra+eng"                 # Tesseract 語言
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 100
CACHE_PATH = "catalog_index.pkl"                 # 快取檔名（放在專案根目錄）

# =========================
# 嘗試自動設定 Tesseract 路徑（Windows 常見安裝路徑）
# =========================
def _maybe_set_tesseract_path():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tessdata_dir = os.path.join(os.path.dirname(path), "tessdata")
            if os.path.isdir(tessdata_dir):
                os.environ["TESSDATA_PREFIX"] = tessdata_dir
            break
_maybe_set_tesseract_path()

# =========================
# 嵌入與 rerank
# =========================
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL_NAME) if USE_RERANK else None

def embed_passages(texts: List[str]) -> np.ndarray:
    return embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)

def embed_query(q: str) -> np.ndarray:
    return embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

# =========================
# 查詢同義詞正規化（可自行擴充）
# =========================
SYNONYMS = {
    "暖白": "3000K", "黃光": "3000K",
    "自然光": "4000K", "白光": "5000K",
    "流明": "lm", "亮度": "lm", "功率": "W", "瓦數": "W",
    "防水": "IP", "顯色": "CRI", "顯色指數": "CRI",
    "角度": "beam", "光束角": "beam",
    "軌道燈": "track light", "投光燈": "flood light",
    "崁燈": "downlight", "投射燈": "flood light"
}
def normalize_query(q: str) -> str:
    out = q
    for k, v in SYNONYMS.items():
        out = out.replace(k, v)
    return out

# =========================
# OCR 與抽取（含強制 OCR、提高 DPI 與 Tesseract config）
# =========================
def _visible_char_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s))

def ocr_pdf_page(fitz_page, dpi: int = 400, lang: str = DEFAULT_OCR_LANG) -> str:
    mat = fitz.Matrix(dpi/72.0, dpi/72.0)
    pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    config = "--psm 6 --oem 3"  # 適合段落/表格小字
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def extract_pdf_chunks(pdf_path: str,
                       catalog_name: str,
                       use_ocr_fallback: bool = True,
                       ocr_lang: str = DEFAULT_OCR_LANG,
                       force_ocr: bool = False,
                       max_chars: int = CHUNK_MAX_CHARS,
                       overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
            for i, page in enumerate(pdf.pages, start=1):
                text = ""
                # 非強制時先試文字層
                if not force_ocr:
                    try:
                        text = page.extract_text() or ""
                        text = re.sub(r"[ \t]+", " ", text)
                        text = re.sub(r"\n{2,}", "\n", text).strip()
                    except Exception:
                        text = ""

                need_ocr = force_ocr
                if not force_ocr:
                    if _visible_char_count(text) < 20 and use_ocr_fallback:
                        need_ocr = True

                if need_ocr:
                    try:
                        text = ocr_pdf_page(doc[i-1], dpi=400, lang=ocr_lang)
                    except Exception:
                        text = text  # 可能仍為空

                if not text:
                    continue

                # 切塊
                start = 0
                while start < len(text):
                    end = min(start + max_chars, len(text))
                    piece = text[start:end]
                    chunks.append({"catalog": catalog_name, "page": i, "text": piece})
                    if end == len(text):
                        break
                    start = max(0, end - overlap)
    except Exception:
        return []
    return chunks

# =========================
# In-memory 索引
# =========================
class InMemoryIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self.embs: np.ndarray | None = None
        self.built = False

    def reset(self):
        self.docs, self.embs, self.built = [], None, False

    def add_docs(self, docs: List[Dict[str, Any]]):
        self.docs.extend(docs)

    def build(self):
        if not self.docs:
            self.embs, self.built = None, False
            return "沒有可建立索引的文件"
        texts = [d["text"] for d in self.docs]
        self.embs = embed_passages(texts)
        self.built = True
        return f"索引建立完成，共 {len(self.docs)} 個片段"

    def search(self, query: str, top_k: int = 8) -> List[Tuple[int, float]]:
        if not self.built or self.embs is None:
            return []
        q = normalize_query(query)
        q_emb = embed_query(q)
        sims = (self.embs @ q_emb)

        k = min(max(top_k * 5, top_k), len(sims))
        cand_idx = np.argpartition(-sims, range(k))[:k]
        pairs = [(int(i), float(sims[i])) for i in cand_idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:max(top_k, 1)]

        if USE_RERANK and reranker is not None and len(pairs) > 0:
            q_dup = [q] * len(pairs)
            cand_texts = [self.docs[i]["text"] for i, _ in pairs]
            scores = reranker.predict(list(zip(q_dup, cand_texts)))
            reranked = list(zip([p[0] for p in pairs], scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            pairs = reranked[:top_k]
        return pairs

INDEX = InMemoryIndex()

# =========================
# 快取：儲存 / 載入 / 清除
# =========================
def save_cache(index: InMemoryIndex) -> str:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(index, f)
    return f"💾 索引已儲存至 {os.path.abspath(CACHE_PATH)}"

def load_cache() -> str:
    global INDEX
    if not os.path.exists(CACHE_PATH):
        return "⚠️ 找不到快取檔，請先建立索引。"
    with open(CACHE_PATH, "rb") as f:
        INDEX = pickle.load(f)
    return f"✅ 已載入快取，可直接查詢（片段：{len(INDEX.docs)}）"

def clear_cache() -> str:
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        return "🧹 已刪除快取檔 catalog_index.pkl"
    return "ℹ️ 沒有可刪除的快取檔"

# =========================
# 建索引（支援實時進度回報）
# =========================
def list_catalog_pdfs() -> str:
    cwd = os.getcwd()
    folder = os.path.join(cwd, "catalogs")
    if not os.path.isdir(folder):
        return f"⚠️ 找不到資料夾：{folder}"
    files = []
    for root, _, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(".pdf"):
                files.append(os.path.join(root, f))
    if not files:
        return f"⚠️ {folder} 內沒有 .pdf 檔案"
    out = [f"🔎 目前工作路徑：{cwd}", f"📃 找到 {len(files)} 本 PDF："]
    out += [" - " + p for p in files]
    return "\n".join(out)

def build_index_from_folder(ocr_lang: str, use_ocr: bool, force_ocr: bool, progress=gr.Progress(track_tqdm=True)):
    try:
        INDEX.reset()
        cwd = os.getcwd()
        folder = os.path.join(cwd, "catalogs")
        if not os.path.isdir(folder):
            yield f"⚠️ 找不到資料夾：{folder}\n請建立 catalogs/ 並放入 PDF。"
            return

        # 掃描 PDF
        pdf_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
        if not pdf_files:
            yield f"⚠️ 在 {folder} 沒找到任何 PDF。"
            return

        yield f"🔎 工作路徑：{cwd}\n📁 共 {len(pdf_files)} 本 PDF：\n" + "\n".join(" - "+p for p in pdf_files)

        total_chunks = 0
        for fi, pdf_path in enumerate(pdf_files, 1):
            base = os.path.basename(pdf_path)
            yield f"📄 [{fi}/{len(pdf_files)}] 開始處理：{base}"
            try:
                with fitz.open(pdf_path) as _doc:
                    if _doc.needs_pass:
                        yield f"❌ 檔案加密需密碼：{pdf_path}"
                        continue
                    n_pages = len(_doc)

                progress(0, desc=f"OCR/抽取 {base}...")
                chunks = extract_pdf_chunks(
                    pdf_path,
                    catalog_name=os.path.splitext(base)[0],
                    use_ocr_fallback=use_ocr,
                    ocr_lang=ocr_lang,
                    force_ocr=force_ocr
                )
                total_chunks += len(chunks)
                INDEX.add_docs(chunks)
                yield f"✅ {base} 加入片段：{len(chunks)}（頁數：{n_pages}）"
            except Exception as e:
                yield f"❌ 解析失敗 {base} → {type(e).__name__}: {e}"

        if not INDEX.docs:
            yield "⚠️ 沒有成功加入任何片段，請檢查 OCR 設定或勾『強制 OCR』再試。"
            return

        build_msg = INDEX.build()
        yield f"🎉 {build_msg}（總片段：{total_chunks}）"
        # 成功後自動儲存快取
        yield save_cache(INDEX)

    except Exception as e:
        yield f"💥 發生錯誤：{type(e).__name__}: {e}"

# =========================
# 查詢
# =========================
def ask(query: str, top_k: int = 6) -> pd.DataFrame:
    if not INDEX.built:
        return pd.DataFrame([{"提示": "請先建立索引，或點『🔄 重新載入快取』"}])
    hits = INDEX.search(query, top_k=top_k)
    rows = []
    for idx, score in hits:
        d = INDEX.docs[idx]
        txt = d["text"].replace("\n", " ")
        if len(txt) > 220:
            txt = txt[:220] + "..."
        rows.append({
            "型錄": d["catalog"],
            "頁碼": d["page"],
            "相似度/分數": round(float(score), 4),
            "片段摘要": txt
        })
    return pd.DataFrame(rows)

def sample_queries():
    return (
        "崁燈 12W 3000K CRI90 24度",
        "戶外 IP65 投光燈 5000K 感應",
        "展示櫃 downlight 15~24 度 不眩光"
    )

# =========================
# Gradio 介面
# =========================
with gr.Blocks(title="Lighting Catalog RAG – OCR + 自動載入 + 快取") as demo:
    gr.Markdown(
        "# 💡 Lighting Catalog RAG – OCR + 自動載入 + 快取\n"
        "把 PDF 放在專案根目的 **catalogs/** → 按「掃描並建立索引」→ 查詢。\n"
        "首次完成後會自動儲存索引到 `catalog_index.pkl`；下次啟動會自動載入。"
    )

    with gr.Row():
        ocr_lang = gr.Dropdown(choices=["chi_tra+eng", "chi_sim+eng", "eng"], value=DEFAULT_OCR_LANG, label="OCR 語言（Tesseract）")
        use_ocr = gr.Checkbox(value=True, label="需要時自動使用 OCR（處理圖片文字）")
        force_ocr = gr.Checkbox(value=False, label="強制 OCR（每頁都 OCR）")

    with gr.Row():
        build_btn = gr.Button("① 掃描 catalogs 並建立索引", scale=3)
        peek_btn = gr.Button("🔍 檢視 catalogs 內容", scale=1)
        reload_btn = gr.Button("🔄 重新載入快取", scale=1)
        clear_btn = gr.Button("🧹 清除快取", scale=1)

    build_status = gr.Markdown("尚未建立")
    build_btn.click(build_index_from_folder, inputs=[ocr_lang, use_ocr, force_ocr], outputs=[build_status])
    peek_btn.click(list_catalog_pdfs, outputs=[build_status])
    reload_btn.click(load_cache, outputs=[build_status])
    clear_btn.click(clear_cache, outputs=[build_status])

    gr.Markdown("## ② 查詢")
    with gr.Row():
        q = gr.Textbox(label="輸入需求（例：『崁燈 12W 3000K CRI90 24度』）", lines=2)
        topk = gr.Slider(1, 15, value=6, step=1, label="回傳數量 Top-K")
    ask_btn = gr.Button("搜尋")
    results = gr.Dataframe(headers=["型錄", "頁碼", "相似度/分數", "片段摘要"], wrap=True)
    ask_btn.click(ask, inputs=[q, topk], outputs=[results])

    with gr.Accordion("📎 查詢範例", open=False):
        ex1, ex2, ex3 = sample_queries()
        gr.Examples(examples=[[ex1], [ex2], [ex3]], inputs=[q], label="點一下填入查詢")

# =========================
# 入口
# =========================
if __name__ == "__main__":
    # 若你安裝在非預設路徑，可手動指定：
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

    # 啟動時自動載入快取（若存在）
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                INDEX = pickle.load(f)
            print(f"✅ 已自動載入快取（片段：{len(INDEX.docs)}）。可直接查詢。")
        except Exception as e:
            print(f"⚠️ 快取載入失敗：{type(e).__name__}: {e}")

    demo.launch()
