# =============================================
# Lighting Catalog — Excel (COM→PDF→OCR) Ultimate
# =============================================

import os, io, re, time, ctypes, pickle, tempfile, shutil, warnings, logging
from typing import List, Dict, Any, Tuple
import numpy as np
import gradio as gr
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------- 基本設定 --------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
USE_RERANK = True

DEFAULT_OCR_LANG = "chi_tra+eng"
CACHE_PATH = "catalog_index.pkl"
CATALOG_DIR = "catalogs"

# OCR/渲染設定
PAGE_DPI = 500               # 較高 DPI，提升 OCR
DEBUG_SAVE = True            # 匯出中間 PDF/PNG 供檢查
DEBUG_DIR  = "debug_out"

os.makedirs(DEBUG_DIR, exist_ok=True)
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# -------------------- Tesseract 路徑 --------------------
def _maybe_set_tesseract_path():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    found = False
    for p in candidates:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            td = os.path.join(os.path.dirname(p), "tessdata")
            if os.path.isdir(td):
                os.environ["TESSDATA_PREFIX"] = td
            print(f"✅ Tesseract: {p}")
            found = True
            break
    if not found:
        fixed = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(fixed):
            pytesseract.pytesseract.tesseract_cmd = fixed
            os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
            print("✅ 使用固定路徑設定 Tesseract。")
        else:
            print("⚠️ 未找到 Tesseract，請確認是否已安裝。")

_maybe_set_tesseract_path()

# -------------------- 模型 --------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker    = CrossEncoder(RERANK_MODEL_NAME) if USE_RERANK else None

def embed_passages(texts): return embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
def embed_query(q):        return embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

# -------------------- OCR 工具 --------------------
def _ocr_image(img: Image.Image, lang=DEFAULT_OCR_LANG) -> str:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(3))
    txt = pytesseract.image_to_string(g, lang=lang, config="--psm 6 --oem 3")
    txt = re.sub(r"[ \t]+", " ", txt)
    return re.sub(r"\n{2,}", "\n", txt).strip()

def _ocr_pdf_file(pdf_path: str, lang=DEFAULT_OCR_LANG) -> Tuple[str, list]:
    """
    固定回傳：(全文字串, 每頁字元數list)
    並將中間 PNG 存到 debug_out，DPI 依 PAGE_DPI
    """
    parts, per_page = [], []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, 1):
                mat = fitz.Matrix(PAGE_DPI/72.0, PAGE_DPI/72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png = pix.tobytes("png")
                if DEBUG_SAVE:
                    out_png = os.path.join(DEBUG_DIR, f"{os.path.basename(pdf_path)}_p{i}.png")
                    with open(out_png, "wb") as f:
                        f.write(png)
                img = Image.open(io.BytesIO(png)).convert("RGB")
                txt = _ocr_image(img, lang=lang)
                if not txt.strip():
                    txt = _ocr_image(img, lang="eng")
                parts.append(txt)
                per_page.append(len(txt))
    except Exception as e:
        print(f"[OCR] 讀取 PDF 失敗：{type(e).__name__}: {e}")
        return "", []
    text = "\n".join([t for t in parts if t.strip()])
    return text, per_page

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip().strip(".")
    return name[:80] or "sheet"

def _short_path(p: str) -> str:
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        buf = ctypes.create_unicode_buffer(1024)
        res = GetShortPathNameW(p, buf, 1024)
        return buf.value if res else p
    except Exception:
        return p

# -------------------- COM 渲染（每分頁獨立 Excel 行程） --------------------
def export_workbook_sheets_to_text(xlsx_path: str, lang=DEFAULT_OCR_LANG) -> Dict[str, str]:
    """
    Excel COM（超強健版）
    - 先把原檔複製到純 ASCII 的暫存資料夾 root/src.xlsx
    - 先用一次性 Excel 行程列出所有分頁名稱
    - 每個分頁各自啟動獨立 Excel 行程：
        sheet.Copy() → 新活頁簿 → SaveAs(.xlsx，失敗退 .xls) → ExportAsFixedFormat(PDF)
    - 全程使用 8.3 短路徑；PDF 複製到 debug_out/ 後做 OCR
    """
    texts: Dict[str, str] = {}
    try:
        import win32com.client, pythoncom
    except Exception:
        print("[COM] pywin32 未安裝或非 Windows。")
        return texts

    # 建立純 ASCII 的工作資料夾
    root = tempfile.mkdtemp(prefix="xls2pdf_job_")
    ascii_root = _short_path(root)
    src_copy = os.path.join(ascii_root, "src.xlsx")
    try:
        shutil.copy2(xlsx_path, src_copy)
    except Exception as e:
        print(f"[COM] 複製檔案失敗：{e}")
        shutil.rmtree(root, ignore_errors=True)
        return texts

    # --- 先列出分頁名稱（單次行程） ---
    def list_sheet_names(src_copy_ascii: str) -> List[str]:
        names = []
        pythoncom.CoInitialize()
        excel = None
        try:
            excel = win32com.client.DispatchEx("Excel.Application")
            excel.Visible = False; excel.DisplayAlerts = False
            try: excel.AutomationSecurity = 1
            except Exception: pass
            wb = excel.Workbooks.Open(_short_path(os.path.abspath(src_copy_ascii)), ReadOnly=True)
            for s in wb.Sheets:
                names.append(str(s.Name))
            wb.Close(SaveChanges=False); excel.Quit(); time.sleep(0.2)
        except Exception as e:
            print(f"[COM] 讀取分頁名稱失敗：{type(e).__name__}: {e}")
            try:
                if excel: excel.Quit()
            except Exception:
                pass
        finally:
            pythoncom.CoUninitialize()
        return names

    sheet_names = list_sheet_names(src_copy)
    if not sheet_names:
        print("[COM] 找不到任何分頁或讀取分頁名稱失敗。")
        shutil.rmtree(root, ignore_errors=True)
        return texts

    # --- 每分頁獨立處理 ---
    def process_one_sheet(src_copy_ascii: str, sheet_name: str) -> str:
        pythoncom.CoInitialize()
        text = ""
        excel = None
        work_root = tempfile.mkdtemp(prefix="xls2pdf_sheet_")
        ascii_wr = _short_path(work_root)

        tmp_xlsx = os.path.join(ascii_wr, "only_sheet.xlsx")
        tmp_xls  = os.path.join(ascii_wr, "only_sheet.xls")
        tmp_pdf  = os.path.join(ascii_wr, "only_sheet.pdf")

        safe = _sanitize_filename(sheet_name)
        out_pdf = os.path.join(DEBUG_DIR, f"{safe}.pdf")

        for p in (tmp_xlsx, tmp_xls, tmp_pdf, out_pdf):
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass

        try:
            excel = win32com.client.DispatchEx("Excel.Application")
            excel.Visible = False; excel.DisplayAlerts = False
            try: excel.AutomationSecurity = 1
            except Exception: pass

            wb = excel.Workbooks.Open(_short_path(os.path.abspath(src_copy_ascii)), ReadOnly=True)

            target = None
            for s in wb.Sheets:
                if str(s.Name) == sheet_name:
                    target = s; break
            if target is None:
                wb.Close(SaveChanges=False); excel.Quit(); pythoncom.CoUninitialize()
                shutil.rmtree(work_root, ignore_errors=True)
                return ""

            target.Copy()
            time.sleep(0.2)
            tmp_wb = excel.ActiveWorkbook
            if tmp_wb is None:
                time.sleep(0.5)
                tmp_wb = excel.ActiveWorkbook
            if tmp_wb is None:
                wb.Close(SaveChanges=False); excel.Quit(); pythoncom.CoUninitialize()
                shutil.rmtree(work_root, ignore_errors=True)
                return ""

            tmp_ws = tmp_wb.Sheets(1)
            try:
                tmp_ws.PageSetup.PrintArea = ""
                tmp_ws.PageSetup.Zoom = False
                tmp_ws.PageSetup.FitToPagesWide = 1
                tmp_ws.PageSetup.FitToPagesTall = False
                tmp_ws.PageSetup.Orientation = 2  # 2=Landscape(橫向). 如需直向改 1
            except Exception:
                pass

            # 先存檔再匯出 PDF（避免「文件未儲存」）
            saved = False
            for attempt in range(2):
                try:
                    if attempt == 0:
                        tmp_wb.SaveAs(_short_path(tmp_xlsx), FileFormat=51)  # .xlsx
                    else:
                        tmp_wb.SaveAs(_short_path(tmp_xls),  FileFormat=56)  # .xls
                    saved = True; break
                except Exception as e:
                    print(f"[COM] SaveAs 失敗（{sheet_name}，attempt={attempt}）：{e}")
                    time.sleep(0.3)

            if not saved:
                try: tmp_wb.Close(SaveChanges=False)
                except Exception: pass
                wb.Close(SaveChanges=False); excel.Quit(); pythoncom.CoUninitialize()
                shutil.rmtree(work_root, ignore_errors=True)
                return ""

            # 匯出 PDF（短路徑）
            exported = False
            for attempt in range(2):
                try:
                    tmp_wb.ExportAsFixedFormat(0, _short_path(tmp_pdf))
                    exported = os.path.exists(tmp_pdf)
                    if exported: break
                except Exception as e:
                    print(f"[COM] Export PDF 失敗（{sheet_name}，attempt={attempt}）：{e}")
                    time.sleep(0.3)

            try: tmp_wb.Close(SaveChanges=False)
            except Exception: pass
            try: wb.Close(SaveChanges=False)
            except Exception: pass
            try: excel.Quit()
            except Exception: pass
            pythoncom.CoUninitialize()

            if exported:
                try: shutil.copy2(tmp_pdf, out_pdf)
                except Exception: out_pdf = tmp_pdf

                ocr_res = _ocr_pdf_file(out_pdf, lang=lang)
                if isinstance(ocr_res, tuple):
                    text = ocr_res[0] if len(ocr_res) >= 1 else ""
                    per_page = ocr_res[1] if len(ocr_res) >= 2 else []
                else:
                    text = ocr_res; per_page = []
                print(f"[OCR] {sheet_name} → 頁數:{len(per_page)}；每頁字元:{per_page}")

        except Exception as e:
            print(f"[COM] 單分頁處理失敗（{sheet_name}）：{type(e).__name__}: {e}")
            try:
                if excel: excel.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        return text

    for name in sheet_names:
        txt = process_one_sheet(src_copy, name)
        if txt.strip():
            texts[name] = txt

    shutil.rmtree(root, ignore_errors=True)
    return texts

# -------------------- 索引 --------------------
class InMemoryIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self.embs = None
        self.built = False

    def reset(self): self.docs, self.embs, self.built = [], None, False
    def add_docs(self, docs): self.docs.extend(docs)

    def build(self):
        if not self.docs: return "⚠️ 沒有內容可建立索引"
        self.embs = embed_passages([d["text"] for d in self.docs])
        self.built = True
        return f"✅ 索引完成：{len(self.docs)} 片段"

    def search(self, q: str, k: int = 6):
        if not self.built or self.embs is None: return []
        q_emb = embed_query(q)
        sims = self.embs @ q_emb
        idx = np.argsort(-sims)[:k]
        pairs = [(int(i), float(sims[int(i)])) for i in idx]
        if USE_RERANK and reranker:
            cand = [self.docs[i]["text"] for i, _ in pairs]
            scores = reranker.predict(list(zip([q]*len(cand), cand)))
            pairs = sorted(zip([p[0] for p in pairs], scores), key=lambda x: x[1], reverse=True)
        return pairs

INDEX = InMemoryIndex()

# -------------------- 快取 --------------------
def save_cache(index):
    with open(CACHE_PATH, "wb") as f: pickle.dump(index, f)
    return f"💾 已儲存索引（{len(index.docs)} 片段）"

def load_cache():
    global INDEX
    if not os.path.exists(CACHE_PATH): return "⚠️ 沒有快取檔"
    with open(CACHE_PATH, "rb") as f: INDEX = pickle.load(f)
    return f"✅ 已載入快取（{len(INDEX.docs)} 片段）"

def clear_cache():
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH); return "🧹 已刪除快取"
    return "ℹ️ 沒有快取可刪除"

# -------------------- 建索引（COM-only） --------------------
def list_excels() -> str:
    folder = os.path.join(os.getcwd(), CATALOG_DIR)
    if not os.path.isdir(folder): return f"⚠️ 找不到資料夾：{folder}"
    files = []
    for f in os.listdir(folder):
        fl = f.lower()
        if not fl.endswith((".xlsx", ".xlsm")): continue
        if f.startswith("~$"): continue  # 跳過 Excel 鎖檔
        files.append(f)
    if not files: return f"⚠️ {folder} 內沒有 Excel 檔"
    return "📊 將處理下列檔案：\n" + "\n".join(f" - {x}" for x in files)

def build_index_from_excel(ocr_lang: str, progress=gr.Progress(track_tqdm=True)):
    try:
        INDEX.reset()
        folder = os.path.join(os.getcwd(), CATALOG_DIR)
        if not os.path.isdir(folder):
            yield f"⚠️ 找不到 {folder}"; return

        xlsx_files = []
        for f in os.listdir(folder):
            if f.startswith("~$"):   # 跳過暫存鎖檔
                continue
            if f.lower().endswith((".xlsx", ".xlsm")):
                xlsx_files.append(os.path.join(folder, f))

        if not xlsx_files:
            yield "⚠️ 未找到任何 Excel 檔"; return

        for i, path in enumerate(xlsx_files, 1):
            base = os.path.basename(path)
            yield f"[{i}/{len(xlsx_files)}] 解析 {base} ..."
            sheet_texts = export_workbook_sheets_to_text(path, lang=ocr_lang)
            added = 0
            for sheet_name, full_text in sheet_texts.items():
                # 切塊
                MAX = 800; OVER = 120
                s = 0
                while s < len(full_text):
                    e = min(s + MAX, len(full_text))
                    INDEX.add_docs([{"catalog": sheet_name, "text": full_text[s:e]}])
                    added += 1
                    if e == len(full_text): break
                    s = max(0, e - OVER)
            yield f"   ↳ 加入片段：{added}"

        if not INDEX.docs:
            yield "⚠️ 沒有成功讀取任何文字。\n請確認：\n- Excel 可手動『另存 PDF』看到內容\n- Tesseract 安裝完整（包含 chi_tra 語言包）\n- debug_out/ 是否有 PDF 與 PNG（若沒有，表示 COM 未成功匯出）"
            return

        yield INDEX.build()
        yield save_cache(INDEX)
    except Exception as e:
        yield f"💥 錯誤：{type(e).__name__}: {e}"

# -------------------- 查詢（Markdown 條列） --------------------
def ask(query: str, top_k: int = 6) -> str:
    if not INDEX.built: return "⚠️ 請先建立索引或載入快取。"
    hits = INDEX.search(query, top_k)
    if not hits: return "❌ 找不到相符內容。"

    out = []
    for idx, _ in hits:
        d = INDEX.docs[idx]
        text = d["text"].strip()
        # 簡易列點化（你可再加規格抽取）
        lines = [x.strip(" ・-••") for x in re.split(r"[\n。•\-]", text) if len(x.strip()) > 2]
        bullets = "\n".join(f"• {x}" for x in lines[:12])
        out.append(f"### {d['catalog']}\n{bullets}")
    return "\n\n".join(out)

# -------------------- 環境檢查 --------------------
def run_diagnostics(ocr_lang: str) -> str:
    report = []

    # 1) Tesseract smoke test
    try:
        img = Image.new("RGB", (800, 200), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "TEST 測試 123 CRI90 3000K IP65 24°", fill="black")
        test_txt = _ocr_image(img, lang=ocr_lang) or _ocr_image(img, lang="eng")
        report.append(f"🧪 Tesseract 測試字數：{len(test_txt)}（內容：{test_txt[:40]}...）")
    except Exception as e:
        report.append(f"🧪 Tesseract 測試失敗：{e}")

    # 2) 列出 Excel
    report.append(list_excels())

    # 3) 挑第一個 Excel 的第一個分頁做 COM→PDF→OCR
    folder = os.path.join(os.getcwd(), CATALOG_DIR)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".xlsx", ".xlsm")) and not f.startswith("~$")] if os.path.isdir(folder) else []
    if not files:
        report.append("⚠️ 沒有 Excel 可測。")
        return "\n\n".join(report)

    x_path = os.path.join(folder, files[0])
    report.append(f"🔧 測試檔：{files[0]} （嘗試全部分頁，回報第一個成功分頁）")

    sheet_texts = export_workbook_sheets_to_text(x_path, lang=ocr_lang)
    if not sheet_texts:
        report.append("❌ COM 匯出沒有任何分頁成功（請看 debug_out/ 是否有 PDF）。")
        return "\n\n".join(report)

    first_sheet = next(iter(sheet_texts.keys()))
    text = sheet_texts[first_sheet]
    report.append(f"✅ 分頁：{first_sheet}，全文字數：{len(text)}")
    report.append(f"📂 請查看 debug_out/ 內的 PDF 與 PNG 是否符合畫面")

    return "\n\n".join(report)

# -------------------- UI --------------------
with gr.Blocks(title="Lighting Catalog — Excel (COM→PDF→OCR) + Diagnostics") as demo:
    gr.Markdown(
        "# 💡 Lighting Catalog — Excel (COM→PDF→OCR)\n"
        f"- Excel 檔放到 `./{CATALOG_DIR}/`（請關閉檔案避免 `~$` 鎖檔）\n"
        "- 本版 **不使用 openpyxl**，只走 **Excel COM → PDF → OCR**，支援浮動圖形/WMF 等\n"
        f"- 中間產物輸出到 `./{DEBUG_DIR}/`，可肉眼檢查\n"
        "- 嵌入：BAAI/bge-m3，Rerank：BAAI/bge-reranker-base"
    )

    with gr.Row():
        ocr_lang = gr.Dropdown(["chi_tra+eng", "chi_sim+eng", "eng"], value=DEFAULT_OCR_LANG, label="OCR 語言")
        build_btn  = gr.Button("① 掃描 Excel 並建立索引", scale=3)
        diag_btn   = gr.Button("🔧 環境檢查", scale=1)
        reload_btn = gr.Button("🔄 載入快取", scale=1)
        clear_btn  = gr.Button("🧹 清除快取", scale=1)

    status = gr.Markdown("尚未建立")
    build_btn.click(build_index_from_excel, inputs=[ocr_lang], outputs=[status])
    diag_btn.click(run_diagnostics, inputs=[ocr_lang], outputs=[status])
    reload_btn.click(load_cache, outputs=[status])
    clear_btn.click(clear_cache, outputs=[status])

    gr.Markdown("## 🔎 查詢")
    with gr.Row():
        q = gr.Textbox(label="輸入需求（例：崁燈 12W 3000K CRI90）", lines=2)
        topk = gr.Slider(1, 12, value=6, step=1, label="顯示數量 Top-K")
    ask_btn = gr.Button("搜尋")
    md_out  = gr.Markdown()
    ask_btn.click(ask, inputs=[q, topk], outputs=[md_out])

if __name__ == "__main__":
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                INDEX = pickle.load(f)
            print(f"✅ 自動載入快取（{len(INDEX.docs)} 片段）")
        except Exception as e:
            print(f"⚠️ 快取載入失敗：{type(e).__name__}: {e}")
    demo.launch()
