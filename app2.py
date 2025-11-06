# app.py — Lighting Catalog RAG（PDF → OCR/抽取 → Cluster → 嵌入 → 檢索）+ LLM 查詢規劃
# 依賴：
#   pip install -U gradio pdfplumber pymupdf pillow pytesseract sentence-transformers numpy python-dotenv openai regex

import os, io, re, time, json, logging, warnings, hashlib, pickle, pathlib
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pdfplumber, fitz  # PyMuPDF
import pytesseract
import gradio as gr
import regex as rxx

from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv, find_dotenv

# -----------------------------
# 靜音一些不是錯誤的訊息
# -----------------------------
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="fitz")

# -----------------------------
# 基本設定
# -----------------------------
EMBED_MODEL_NAME   = "BAAI/bge-m3"                 # 多語向量
RERANK_MODEL_NAME  = "BAAI/bge-reranker-base"      # 交叉編碼 rerank（可關）
USE_RERANK         = True

DEFAULT_OCR_LANG   = "chi_tra+eng"
CHUNK_MAX_CHARS    = 900
CHUNK_OVERLAP      = 120

# 查詢同義詞（可擴充）
SYNONYMS = {
    "暖白":"3000K","黃光":"3000K","自然光":"4000K","白光":"5000K",
    "流明":"lm","亮度":"lm","功率":"W","瓦數":"W",
    "防水":"IP","顯色":"CRI","顯色指數":"CRI",
    "角度":"beam","光束角":"beam",
    "軌道燈":"track light","投光燈":"flood light","崁燈":"downlight","投射燈":"flood light"
}

# ---- 燈款類型字典（可依你的資料擴充）----
LAMP_TYPES = [
    "吸頂燈","崁燈","軌道燈","投光燈","投射燈","壁燈","吊燈","檯燈","立燈",
    "燈帶","日光燈","路燈","草地燈","庭園燈","洗牆燈","天井燈","格柵燈","線型燈"
]

# 同義詞 → 正規化
TYPE_ALIASES = {
    "downlight":"崁燈","嵌燈":"崁燈","崁入燈":"崁燈",
    "ceiling light":"吸頂燈","吸頂":"吸頂燈",
    "track light":"軌道燈","導軌燈":"軌道燈",
    "flood light":"投光燈","泛光燈":"投光燈",
    "spot light":"投射燈","射燈":"投射燈","投射":"投射燈",
    "pendant":"吊燈",
    "wall light":"壁燈",
    "strip":"燈帶","條燈":"燈帶","線燈":"線型燈","線型":"線型燈",
    "high bay":"天井燈","高天井燈":"天井燈",
    "grid":"格柵燈","格柵":"格柵燈",
    "garden":"庭園燈","草地燈":"草地燈",
}

def _normalize_types(types: list[str]) -> list[str]:
    out = []
    for t in types or []:
        if not isinstance(t, str): continue
        t0 = t.strip().lower()
        if not t0: continue
        if t0 in TYPE_ALIASES:
            std = TYPE_ALIASES[t0]
        else:
            std = next((lt for lt in LAMP_TYPES if lt.lower() == t0), None)
            if std is None:
                std = next((lt for lt in LAMP_TYPES if t0 in lt.lower()), None)
        if std and std not in out:
            out.append(std)
    return out

# -----------------------------
# OpenAI 初始化（支援 .env + UI 臨時設定；新/舊 SDK）
# -----------------------------
DOTENV_PATH = find_dotenv(usecwd=True); load_dotenv(DOTENV_PATH or "", override=True)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI = None
OPENAI_STATUS = []

def _init_openai():
    global OPENAI, OPENAI_STATUS
    OPENAI_STATUS.clear()
    if not OPENAI_API_KEY:
        OPENAI = None; OPENAI_STATUS.append("no_key"); return
    try:
        from openai import OpenAI
        OPENAI = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_STATUS.append("new_sdk_ok"); return
    except Exception as e:
        OPENAI_STATUS.append(f"new_sdk_fail:{type(e).__name__}")
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        OPENAI = openai
        OPENAI_STATUS.append("legacy_sdk_ok"); return
    except Exception as e:
        OPENAI = None; OPENAI_STATUS.append(f"legacy_sdk_fail:{type(e).__name__}")

def _chat_once(messages, model="gpt-4o-mini", max_tokens=400, temperature=0.2):
    if OPENAI is None:
        raise RuntimeError("openai_client_not_ready")
    # new sdk
    if hasattr(OPENAI, "chat") and hasattr(OPENAI.chat, "completions"):
        resp = OPENAI.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    # legacy
    import types
    if isinstance(OPENAI, types.ModuleType) and hasattr(OPENAI, "ChatCompletion"):
        resp = OPENAI.ChatCompletion.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp["choices"][0]["message"]["content"].strip()
    raise RuntimeError("unsupported_openai_client")

def set_api_key_runtime(key: str) -> str:
    global OPENAI_API_KEY
    k = (key or "").strip()
    if not k.startswith("sk-"): return "❌ 看起來不像 OpenAI Key（需 sk- 開頭）。"
    os.environ["OPENAI_API_KEY"] = k
    OPENAI_API_KEY = k
    _init_openai()
    return f"✅ 已套用。狀態：{', '.join(OPENAI_STATUS)}"

_init_openai()
print(f"[dotenv] path={DOTENV_PATH or '(none)'} key_loaded={bool(OPENAI_API_KEY)} status={','.join(OPENAI_STATUS)}")

# -----------------------------
# Tesseract 自動偵測（Windows）
# -----------------------------
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

# -----------------------------
# 嵌入與 rerank
# -----------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
embed_model.max_seq_length = 256  # 控制記憶體
reranker   = CrossEncoder(RERANK_MODEL_NAME) if USE_RERANK else None

def embed_passages(texts: List[str]) -> np.ndarray:
    out = []
    bs = 16
    for i in range(0, len(texts), bs):
        chunk = [t[:1200] for t in texts[i:i+bs]]
        vec = embed_model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True, batch_size=bs, show_progress_bar=False)
        out.append(vec)
    return np.vstack(out) if out else np.zeros((0, 1024), dtype=np.float32)

def embed_query(q: str) -> np.ndarray:
    return embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

# -----------------------------
# OCR 與抽取
# -----------------------------
def _visible_char_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s))

def ocr_pdf_page(fitz_page, dpi: int = 400, lang: str = DEFAULT_OCR_LANG) -> str:
    mat = fitz.Matrix(dpi/72.0, dpi/72.0)
    pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def extract_pdf_chunks(pdf_path: str, catalog_name: str,
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
                if not force_ocr:
                    try:
                        text = page.extract_text() or ""
                        text = re.sub(r"[ \t]+", " ", text)
                        text = re.sub(r"\n{2,}", "\n", text).strip()
                    except Exception:
                        text = ""
                need_ocr = force_ocr
                if not force_ocr and _visible_char_count(text) < 20 and use_ocr_fallback:
                    need_ocr = True
                if need_ocr:
                    try:
                        text = ocr_pdf_page(doc[i-1], dpi=400, lang=ocr_lang)
                    except Exception:
                        pass
                if not text:  # 仍空
                    continue
                # 切塊
                start = 0
                while start < len(text):
                    end = min(start + max_chars, len(text))
                    piece = text[start:end]
                    chunks.append({"catalog": catalog_name, "page": i, "text": piece})
                    if end == len(text): break
                    start = max(0, end - overlap)
    except Exception:
        return []
    return chunks

# -----------------------------
# PKL 快取（最小）
# -----------------------------
PKL_CACHE_PATH = ".rag_cache/index.pkl"

def _manifest_for_folder(folder: str) -> dict:
    base = pathlib.Path(folder)
    items = []
    for p in sorted(base.rglob("*.pdf")):
        st = p.stat()
        items.append({"p": str(p.relative_to(base)), "s": st.st_size, "m": int(st.st_mtime)})
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(blob).hexdigest(), "items": items}

def save_cache_pkl(index, folder: str) -> str:
    os.makedirs(os.path.dirname(PKL_CACHE_PATH), exist_ok=True)
    manifest = _manifest_for_folder(folder)
    payload = {
        "manifest": manifest,
        "embed_model": EMBED_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME,
        "embs": index.embs,
        "cluster_vecs": index.cluster_vecs,
        "docs": index.docs,
        "built": index.built,
        "cluster_docs": index.cluster_docs,
        "cluster_built": index.cluster_built,
        "cluster_map": index.cluster_map,
        "doc2cluster": index.doc2cluster,
    }
    with open(PKL_CACHE_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return f"💾 已儲存 PKL 快取：{PKL_CACHE_PATH}"

def load_cache_pkl(index, folder: str) -> str:
    if not os.path.exists(PKL_CACHE_PATH):
        return "ℹ️ 找不到 PKL 快取。"
    try:
        with open(PKL_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return f"⚠️ 載入 PKL 失敗：{type(e).__name__}: {e}"

    cur = _manifest_for_folder(folder)
    old = data.get("manifest", {})
    if not old or old.get("sha256") != cur.get("sha256"):
        return "ℹ️ catalogs 內容已變更，略過舊 PKL 快取。"

    index.docs = data.get("docs", [])
    index.built = data.get("built", False)
    index.embs = data.get("embs", None)

    index.cluster_docs = data.get("cluster_docs", [])
    index.cluster_built = data.get("cluster_built", False)
    index.cluster_vecs = data.get("cluster_vecs", None)

    index.cluster_map = data.get("cluster_map", {})
    index.doc2cluster = data.get("doc2cluster", {})

    return "✅ 已載入 PKL 快取（資料未變動）"

# -----------------------------
# In-memory 索引（原功能 + 擴充 Cluster）
# -----------------------------
class InMemoryIndex:
    def __init__(self):
        # chunk-level
        self.docs: List[Dict[str, Any]] = []
        self.embs: np.ndarray | None = None
        self.built = False
        # cluster-level
        self.cluster_docs: List[Dict[str, Any]] = []
        self.cluster_vecs: np.ndarray | None = None
        self.cluster_built = False
        self.cluster_map = {}   # cluster_id -> {"members":[doc_idx...], "signature": str, "spec": {...}, "product_name": str}
        self.doc2cluster = {}   # doc_idx -> cluster_id

    def reset(self):
        self.docs, self.embs, self.built = [], None, False
        self.cluster_docs, self.cluster_vecs, self.cluster_built = [], None, False
        self.cluster_map.clear()
        self.doc2cluster.clear()

    def add_docs(self, docs: List[Dict[str, Any]]):
        self.docs.extend(docs)

    def build(self):
        if not self.docs:
            self.embs, self.built = None, False
            return "沒有可建立索引的文件"
        texts = [d["text"] for d in self.docs]
        self.embs = embed_passages(texts)
        self.built = True
        dim = self.embs.shape[1] if hasattr(self.embs,"shape") else "?"
        return f"索引建立完成，共 {len(self.docs)} 片段（維度 {dim}）"

    # chunk-level search（保留）
    def search(self, query: str, top_k: int = 8) -> List[Tuple[int, float]]:
        if not self.built or self.embs is None:
            return []
        q = query
        for k,v in SYNONYMS.items(): q = q.replace(k, v)
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
            reranked = list(zip([p[0] for p in pairs], [float(s) for s in scores]))
            reranked.sort(key=lambda x: x[1], reverse=True)
            pairs = reranked[:top_k]
        return pairs

    # --------- Cluster 構建與向量 ---------
    @staticmethod
    def _extract_model_no(text: str) -> str | None:
        m = re.search(r"[A-Z]{2,}[A-Z0-9][A-Z0-9\-_]{1,}", text)
        return m.group(0) if m else None

    @staticmethod
    def _parse_specs(text: str) -> Dict[str, Any]:
        spec = {"watt": None, "k": None, "cri": None, "ip": None, "beam": None, "features": [], "type": None}
        m = re.findall(r"\b(\d{1,3})\s?W\b", text, flags=re.I)
        if m: spec["watt"] = int(m[0])
        m = re.findall(r"\b(\d{3,4})\s?K\b", text, flags=re.I)
        if m: spec["k"] = int(m[0])
        m = re.findall(r"CRI\s*(?:≥|≧)?\s*(\d{2})", text, flags=re.I)
        if m: spec["cri"] = int(m[0])
        m = re.findall(r"\bIP\s?(\d{2})\b", text, flags=re.I)
        if m: spec["ip"] = int(m[0])
        m = re.findall(r"(\d{1,3})\s?(?:°|度)", text, flags=re.I)
        if m: spec["beam"] = int(m[0])
        feats = []
        for kw in ["防水","防潮","防濕","防眩","感應","高CRI","耐候","無頻閃"]:
            if kw in text:
                feats.append(kw)
        spec["features"] = list(dict.fromkeys(feats))
        # 粗略猜測燈款類型
        low = text.lower()
        for k, v in TYPE_ALIASES.items():
            if k in low:
                spec["type"] = v; break
        for t in LAMP_TYPES:
            if t in text:
                spec["type"] = t; break
        return spec

    @staticmethod
    def _spec_signature(spec: Dict[str, Any]) -> str:
        parts = []
        if spec.get("type"): parts.append(spec["type"])
        if spec.get("watt") is not None: parts.append(f"{spec['watt']}W")
        if spec.get("k")    is not None: parts.append(f"{spec['k']}K")
        if spec.get("cri")  is not None: parts.append(f"CRI≥{spec['cri']}")
        if spec.get("ip")   is not None: parts.append(f"IP{spec['ip']}")
        if spec.get("beam") is not None: parts.append(f"{spec['beam']}°")
        if spec.get("features"): parts += spec["features"][:2]
        return " ".join(parts) if parts else ""

    def build_clusters(self):
        if not self.docs:
            self.cluster_docs, self.cluster_vecs, self.cluster_built = [], None, False
            self.cluster_map.clear(); self.doc2cluster.clear()
            return "沒有可建立叢集的文件"

        buckets: Dict[str, List[int]] = {}
        for i, d in enumerate(self.docs):
            t = d.get("text","")
            catalog = d.get("catalog","")
            model = self._extract_model_no(t)
            if model:
                cid = f"{catalog}:{model}"
            else:
                cid = f"{catalog}:page{d.get('page',0)}"
            buckets.setdefault(cid, []).append(i)

        cluster_docs = []
        cluster_map = {}
        doc2cluster = {}
        for cid, idxs in buckets.items():
            texts = [self.docs[i]["text"] for i in idxs]
            catalog = self.docs[idxs[0]].get("catalog","")
            pages = sorted(list(dict.fromkeys([self.docs[i].get("page",0) for i in idxs])))
            merged_text = "\n".join(texts)
            model_no = cid.split(":")[1] if ":" in cid else None
            spec = self._parse_specs(merged_text)
            signature = f"{model_no or 'Unknown'} {self._spec_signature(spec)}".strip()
            product_name = model_no or (spec.get("type") or "Unknown")

            cluster_docs.append({
                "cluster_id": cid,
                "catalog": catalog,
                "model_no": model_no,
                "product_name": product_name,
                "pages": pages,
                "spec": spec,
                "text": f"{signature}\n{merged_text}"
            })
            cluster_map[cid] = {"members": idxs, "signature": signature, "spec": spec, "product_name": product_name}
            for i in idxs:
                doc2cluster[i] = cid

        self.cluster_docs = cluster_docs
        self.cluster_map = cluster_map
        self.doc2cluster = doc2cluster
        return f"建立 {len(self.cluster_docs)} 個叢集（產品單位）"

    def build_cluster_embeddings(self):
        if not self.cluster_docs:
            self.cluster_vecs, self.cluster_built = None, False
            return "沒有叢集可嵌入"
        texts = [c["text"] for c in self.cluster_docs]
        vecs = embed_passages(texts)
        self.cluster_vecs = vecs
        self.cluster_built = True
        return f"叢集向量建立完成，共 {len(self.cluster_docs)} 款產品"

    def _constraints_from_plan(self, plan: dict) -> Dict[str, Any]:
        min_ip = None
        tokens = [*plan.get("must_terms",[]), *plan.get("nice_to_have",[])]
        if any(t for t in tokens if ("防水" in t or "防潮" in t)):
            min_ip = 44
        for t in tokens:
            m = re.search(r"IP\s?(\d{2})", t, flags=re.I)
            if m:
                ip = int(m.group(1))
                min_ip = max(min_ip or 0, ip)
        return {"min_ip": min_ip}

    def search_clusters(self, query: str, top_clusters: int = 10, per_cluster_docs: int = 3) -> Tuple[List[Tuple[str,float]], List[Tuple[int,float]]]:
        if not self.cluster_built or self.cluster_vecs is None:
            return [], []
        q = query
        for k,v in SYNONYMS.items(): q = q.replace(k, v)
        qv = embed_query(q)

        sims = (self.cluster_vecs @ qv)
        k = min(max(top_clusters * 4, top_clusters), len(sims))
        cand_idx = np.argpartition(-sims, range(k))[:k]
        cluster_pairs = [(int(i), float(sims[i])) for i in cand_idx]
        cluster_pairs.sort(key=lambda x: x[1], reverse=True)
        cluster_pairs = cluster_pairs[:top_clusters]

        doc_pairs: List[Tuple[int,float]] = []
        if self.built and self.embs is not None:
            for ci, _ in cluster_pairs:
                cid = self.cluster_docs[ci]["cluster_id"]
                idxs = self.cluster_map.get(cid,{}).get("members",[])
                sims_d = [(i, float(self.embs[i] @ qv)) for i in idxs]
                sims_d.sort(key=lambda x: x[1], reverse=True)
                doc_pairs.extend(sims_d[:per_cluster_docs])

        if USE_RERANK and reranker is not None and doc_pairs:
            q_dup = [q]*len(doc_pairs)
            texts = [self.docs[i]["text"] for i,_ in doc_pairs]
            scores = reranker.predict(list(zip(q_dup, texts)))
            doc_pairs = [(doc_pairs[i][0], float(scores[i])) for i in range(len(doc_pairs))]
            doc_pairs.sort(key=lambda x:x[1], reverse=True)

        cluster_scores: Dict[str, List[float]] = {}
        for i, s in doc_pairs:
            cid = self.doc2cluster.get(i)
            if not cid: continue
            cluster_scores.setdefault(cid, []).append(s)
        ranked = sorted(
            [(cid, max(v)) for cid, v in cluster_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        return ranked, doc_pairs

    def search_clusters_with_constraints(self, query: str, plan: dict, top_clusters: int = 10) -> List[Tuple[str,float]]:
        if not self.cluster_built or self.cluster_vecs is None:
            return []
        q = query
        for k,v in SYNONYMS.items(): q = q.replace(k, v)
        qv = embed_query(q)

        cons = self._constraints_from_plan(plan)
        min_ip = cons.get("min_ip")

        candidates = []
        for ci, cdoc in enumerate(self.cluster_docs):
            ip = cdoc.get("spec",{}).get("ip")
            if (min_ip is not None) and (ip is not None) and (ip < min_ip):
                continue
            # 類型過濾（若有）
            plan_types = plan.get("types") or []
            ctype = cdoc.get("spec",{}).get("type")
            if plan_types and ctype and (ctype not in plan_types):
                pass
            score = float(self.cluster_vecs[ci] @ qv)
            candidates.append((ci, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_clusters]

        if USE_RERANK and reranker is not None and candidates:
            q_dup = [q]*len(candidates)
            texts = [self.cluster_docs[ci]["text"] for ci,_ in candidates]
            scores = reranker.predict(list(zip(q_dup, texts)))
            candidates = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
            candidates.sort(key=lambda x:x[1], reverse=True)

        out = []
        for ci, sc in candidates:
            out.append((self.cluster_docs[ci]["cluster_id"], float(sc)))
        return out

INDEX = InMemoryIndex()

# -----------------------------
# 可用類型偵測（供 LLM 提示）
# -----------------------------
def infer_available_types_from_index() -> list[str]:
    if not INDEX.cluster_docs:
        return LAMP_TYPES
    text_all = " \n".join(c["text"] for c in INDEX.cluster_docs if c.get("text"))
    found = []
    low = text_all.lower()
    for t in LAMP_TYPES:
        if (t in text_all) or (t.lower() in low):
            found.append(t)
    return found or LAMP_TYPES

# -----------------------------
# 建索引（掃描 catalogs/，OCR 抽取，嵌入 + Cluster 架構）
# -----------------------------
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

        pdf_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
        if not pdf_files:
            yield f"⚠️ 在 {folder} 沒找到任何 PDF。"; return

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
            yield "⚠️ 沒有成功加入任何片段，請檢查 OCR 設定或勾『強制 OCR』再試。"; return

        build_msg = INDEX.build()
        yield f"🔧 {build_msg}（總片段：{total_chunks}）"

        cmsg = INDEX.build_clusters()
        yield f"🧩 {cmsg}"
        cemsg = INDEX.build_cluster_embeddings()
        yield f"🎯 {cemsg}"

        yield f"🎉 完成：chunk 片段 {len(INDEX.docs)}、產品叢集 {len(INDEX.cluster_docs)}。"

        # 自動存 PKL
        yield save_cache_pkl(INDEX, folder)

    except Exception as e:
        yield f"💥 發生錯誤：{type(e).__name__}: {e}"

# -----------------------------
# LLM 查詢規劃（含產品名/型號/類型）
# -----------------------------
_RULE_HINTS = [
    (rxx.compile(r"浴室|衛浴|潮濕|廁所"),        ["IP65","IP54","防水","防潮","防濕"]),
    (rxx.compile(r"戶外|庭院|陽台|雨"),          ["IP65","IP66","防水","耐候"]),
    (rxx.compile(r"廚房|油煙"),                  ["易清潔","防油汙","IP44"]),
    (rxx.compile(r"走道|玄關|樓梯"),             ["感應","微波感應","廣角","防眩"]),
    (rxx.compile(r"展示櫃|展櫃|陳列"),           ["窄角","15°","24°","高CRI","Ra≥90"]),
    (rxx.compile(r"閱讀|辦公|書房"),             ["4000K","5000K","防眩","無頻閃"]),
    (rxx.compile(r"攝影|商品拍攝"),              ["高CRI","Ra≥90","Ra≥95","高流明"]),
]

def _rule_plan(user_text:str)->dict:
    q = user_text.strip()
    must, nice, types = [], [], []

    for pat,kws in _RULE_HINTS:
        if pat.search(q): nice.extend(kws)

    must += rxx.findall(r"\b\d{2,4}K\b", q, flags=rxx.I)
    must += rxx.findall(r"\b\d{1,3}\s?W\b", q, flags=rxx.I)
    must += rxx.findall(r"\bIP\s?\d{2}\b", q, flags=rxx.I)
    must += rxx.findall(r"CRI\s*(?:≥|≧)?\s*\d{2}", q, flags=rxx.I)

    for key, std in TYPE_ALIASES.items():
        if key in q.lower():
            types.append(std)
    for t in LAMP_TYPES:
        if t in q:
            types.append(t)

    must = list(dict.fromkeys([m.replace(" ","") for m in must]))
    nice = list(dict.fromkeys(nice))
    types = _normalize_types(types)

    return {
        "product_names": [],
        "model_numbers": [],
        "types": types,
        "exclude_types": [],
        "must_terms": must,
        "nice_to_have": nice,
        "negations": []
    }

def infer_available_types_from_index() -> list[str]:
    if not INDEX.cluster_docs:
        return LAMP_TYPES
    text_all = " \n".join(c["text"] for c in INDEX.cluster_docs if c.get("text"))
    found = []
    low = text_all.lower()
    for t in LAMP_TYPES:
        if (t in text_all) or (t.lower() in low):
            found.append(t)
    return found or LAMP_TYPES

def llm_plan_query(user_text: str) -> dict:
    if OPENAI is None:
        return _rule_plan(user_text)

    if INDEX.cluster_docs:
        all_models = [c.get("model_no") for c in INDEX.cluster_docs if c.get("model_no")]
        all_names = [c.get("product_name") for c in INDEX.cluster_docs if c.get("product_name")]
        all_candidates = list(dict.fromkeys(all_models + all_names))[:80]
        sample_text = "、".join(all_candidates)
        catalog_hint = f"以下是資料集中出現的產品或型號：{sample_text}。"
    else:
        catalog_hint = ""

    available_types = infer_available_types_from_index()
    types_csv = "、".join(available_types)

    sys = (
        "你是專業燈具顧問與檢索助手。請閱讀使用者需求，並根據提供的候選型號清單與資料，"
        "輸出 JSON，鍵包含：product_names、model_numbers、types、exclude_types、must_terms、nice_to_have、negations。"
        "若使用者輸入包含具體品牌、型號，或與現有 PDF 相符之名稱及單詞，請優先列於 product_names / model_numbers。"
        "如果與現有pdf有直接推薦關聯，請優先列於 product_names / model_numbers。"
        "規格請用標準標記（3000K、20W、CRI≥90、IP65、15°/24°、防水/防潮/防眩/感應/高CRI）。"
        "只回 JSON，不要多餘說明。"
        f"候選燈款類型：{types_csv}。{catalog_hint}"
    )

    fewshot_user = "我想找 IP65 防水的浴室吸頂燈 DL-123"
    fewshot_assistant = json.dumps({
        "product_names": ["DL-123 吸頂燈"],
        "model_numbers": ["DL-123"],
        "types": ["吸頂燈"],
        "exclude_types": [],
        "must_terms": ["IP65", "防水"],
        "nice_to_have": ["防潮", "3000K"],
        "negations": []
    }, ensure_ascii=False)

    usr = f"""使用者需求：{user_text}
請回傳 JSON，例如：
{{
 "product_names":["DL-123 吸頂燈"],
 "model_numbers":["DL-123"],
 "types":["吸頂燈"],
 "exclude_types":[],
 "must_terms":["IP65","防水"],
 "nice_to_have":["防眩","高CRI"],
 "negations":[]
}}"""

    try:
        text = _chat_once(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": fewshot_user},
                {"role": "assistant", "content": fewshot_assistant},
                {"role": "user", "content": usr},
            ],
            model="gpt-4o-mini", max_tokens=250, temperature=0.15
        )
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not dict")

        for k in ("product_names","model_numbers","types","exclude_types",
                  "must_terms","nice_to_have","negations"):
            if k not in data or not isinstance(data[k], list):
                data[k] = []

        data["types"] = [t for t in _normalize_types(data["types"]) if t in available_types][:2]
        data["exclude_types"] = [t for t in _normalize_types(data["exclude_types"]) if t in available_types][:3]
        return data
    except Exception as e:
        print("llm_plan_query fallback:", type(e).__name__, e)
        return _rule_plan(user_text)

def build_search_string(plan: dict) -> str:
    must = plan.get("must_terms", [])
    nice = plan.get("nice_to_have", [])
    types = plan.get("types", [])
    prods = plan.get("product_names", []) + plan.get("model_numbers", [])
    tokens = []
    tokens += prods * 3
    tokens += types * 2
    tokens += must * 2
    tokens += nice
    return " ".join(tokens) if tokens else ""

# -----------------------------
# 查詢（片段層，保留原本）
# -----------------------------
def ask_basic(query: str, top_k: int = 6) -> str:
    if not INDEX.built:
        return "⚠️ 請先建立索引。"
    if not query.strip():
        return "請輸入查詢條件。"
    hits = INDEX.search(query.strip(), top_k=top_k)
    if not hits: return "❌ 找不到相符內容。"
    out=[]
    spec_re = {
        "瓦數":   r"\b(\d{1,3}\s?W)\b",
        "色溫":   r"\b(\d{3,4}\s?K)\b",
        "CRI":    r"(CRI\s*(?:≥|≧)?\s*\d{2})",
        "光束角": r"(\d{1,3}\s?(?:°|度))",
        "IP":     r"\bIP\s?\d{2}\b",
        "流明":   r"\b\d{3,5}\s?lm\b",
        "特性":   r"(防水|防潮|防濕|防眩|感應|高CRI|耐候|無頻閃)"
    }
    for idx,_ in hits:
        d = INDEX.docs[idx]
        sheet = f"{d.get('catalog','頁面')} p.{d.get('page','?')}"
        text  = d.get("text","").replace("\n"," ")
        bullets=[]
        for k,rg in spec_re.items():
            ms = list(re.finditer(rg, text, re.I))
            if ms:
                vals = list(dict.fromkeys([m.group(1) if m.lastindex else m.group(0) for m in ms]))
                bullets.append(f"- **{k}**：{', '.join(vals[:4])}")
        if not bullets:
            bullets.append(f"- 內容：{text[:160]}{'…' if len(text)>160 else ''}")
        out.append(f"### {sheet}\n" + "\n".join(bullets))
    return "\n\n".join(out)

def ask_ai(user_text: str, top_k: int = 6) -> str:
    if not INDEX.built:
        return "⚠️ 請先建立索引。"
    plan = llm_plan_query(user_text)
    q = build_search_string(plan) or user_text
    hits = INDEX.search(q, top_k=top_k)
    if not hits:
        return f"❌ 找不到結果（查詢字串：{q}）"
    header = (
        f"**查詢理解** → products={plan.get('product_names',[])+plan.get('model_numbers',[])}, "
        f"types={plan.get('types',[])}, must={plan.get('must_terms',[])}, nice={plan.get('nice_to_have',[])}"
    )
    return header + "\n\n" + ask_basic(q, top_k)

# -----------------------------
# 產品為中心（Cluster）查詢
# -----------------------------
def _fmt_product_card(cid: str, score: float) -> str:
    meta = INDEX.cluster_map.get(cid, {})
    spec = meta.get("spec", {})
    sig  = meta.get("signature", "")
    cdoc = next((c for c in INDEX.cluster_docs if c["cluster_id"]==cid), None)
    pages = cdoc.get("pages", []) if cdoc else []
    catalog = cdoc.get("catalog","?") if cdoc else "?"
    model_no = cdoc.get("model_no","Unknown") if cdoc else "Unknown"
    product_name = cdoc.get("product_name","Unknown") if cdoc else "Unknown"

    line = []
    if spec.get("type"): line.append(spec["type"])
    if spec.get("watt") is not None: line.append(f"{spec['watt']}W")
    if spec.get("k")    is not None: line.append(f"{spec['k']}K")
    if spec.get("cri")  is not None: line.append(f"CRI≥{spec['cri']}")
    if spec.get("ip")   is not None: line.append(f"IP{spec['ip']}")
    if spec.get("beam") is not None: line.append(f"{spec['beam']}°")
    if spec.get("features"): line += spec["features"][:2]
    spec_line = " / ".join(line) if line else "(無解析到規格)"

    return (
        f"### {product_name} | 型號：{model_no} 〔{catalog}〕  \n"
        f"- **規格**：{spec_line}  \n"
        f"- **頁碼**：{pages[:8]}{'…' if len(pages)>8 else ''}  \n"
        f"- **相似度/分數**：{score:.4f}  \n"
        f"- **摘要**：{sig}\n"
    )

def ask_product(query: str, top_k: int = 6) -> str:
    if not INDEX.cluster_built:
        return "⚠️ 需先建立叢集索引（按『掃描並建立索引』會自動完成）。"
    if not query.strip():
        return "請輸入查詢條件。"
    ranked, _docpairs = INDEX.search_clusters(query.strip(), top_clusters=max(top_k*2, top_k), per_cluster_docs=3)
    if not ranked:
        return "❌ 找不到相符產品。"
    out = []
    for cid, sc in ranked[:top_k]:
        out.append(_fmt_product_card(cid, sc))
    return "\n\n".join(out)

def ask_product_ai(user_text: str, top_k: int = 6) -> str:
    if not INDEX.cluster_built:
        return "⚠️ 需先建立叢集索引（按『掃描並建立索引』會自動完成）。"
    plan = llm_plan_query(user_text)
    q = build_search_string(plan) or user_text
    ranked = INDEX.search_clusters_with_constraints(q, plan, top_clusters=max(top_k*3, top_k))
    if not ranked:
        ranked, _ = INDEX.search_clusters(q, top_clusters=max(top_k*2, top_k), per_cluster_docs=3)
        if not ranked:
            return f"❌ 找不到結果（查詢字串：{q}）"
    header = (
        f"**查詢理解（產品）** → products={plan.get('product_names',[])+plan.get('model_numbers',[])}, "
        f"types={plan.get('types',[])}, must={plan.get('must_terms',[])}, nice={plan.get('nice_to_have',[])}"
    )
    cards = "\n\n".join(_fmt_product_card(cid, sc) for cid, sc in ranked[:top_k])
    return header + "\n\n" + cards

# -----------------------------
# UI + 快捷鍵（只針對 PKL 快取）
# -----------------------------
def _ui_load_cache_pkl():
    folder = os.path.join(os.getcwd(), "catalogs")
    return load_cache_pkl(INDEX, folder)

def _ui_save_cache_pkl():
    folder = os.path.join(os.getcwd(), "catalogs")
    return save_cache_pkl(INDEX, folder)

HOTKEY_JS = """
<script>
(function(){
  const clickById = (id) => { const el = document.getElementById(id); if(el) el.click(); }
  document.addEventListener('keydown', function(e){
    if(!e.ctrlKey) return;
    const k = e.key.toLowerCase();
    if(k==='l'){ e.preventDefault(); clickById('btn_load_pkl'); } // Ctrl+L 載入 PKL
    if(k==='s'){ e.preventDefault(); clickById('btn_save_pkl'); } // Ctrl+S 儲存 PKL
  }, true);
})();
</script>
"""

with gr.Blocks(title="Lighting Catalog RAG – PDF + LLM 查詢規劃（含產品叢集）") as demo:
    gr.HTML(HOTKEY_JS)
    gr.Markdown(
        "# 💡 Lighting Catalog RAG – PDF + LLM 查詢規劃（含產品叢集）\n"
        "- 將 PDF 放在專案根目的 **catalogs/** → 按「掃描並建立索引」。\n"
        "- 可用『關鍵字直接搜』或『AI 理解後搜尋』，以及『**以產品為中心搜尋**』。\n"
        "- **快捷鍵**：Ctrl+L 載入 PKL、Ctrl+S 儲存 PKL。"
    )

    with gr.Row():
        ocr_lang = gr.Dropdown(choices=["chi_tra+eng","chi_sim+eng","eng"], value=DEFAULT_OCR_LANG, label="OCR 語言（Tesseract）")
        use_ocr = gr.Checkbox(value=True, label="需要時自動使用 OCR（圖片頁面）")
        force_ocr = gr.Checkbox(value=False, label="強制 OCR（每頁都 OCR）")

    with gr.Row():
        build_btn = gr.Button("① 掃描 catalogs 並建立索引（含產品叢集）", elem_id="btn_build", scale=2)
        peek_btn = gr.Button("🔍 檢視 catalogs 內容", scale=1)
        key_in = gr.Textbox(label="（可選）臨時設定 OpenAI API Key：sk-xxxx", type="password")
        key_btn = gr.Button("套用Key", scale=1)
        load_pkl_btn = gr.Button("載入 PKL 快取", elem_id="btn_load_pkl", scale=1)
        save_pkl_btn = gr.Button("儲存 PKL 快取", elem_id="btn_save_pkl", scale=1)

    build_status = gr.Markdown("尚未建立")

    build_btn.click(build_index_from_folder, inputs=[ocr_lang, use_ocr, force_ocr], outputs=[build_status])
    peek_btn.click(list_catalog_pdfs, outputs=[build_status])
    key_btn.click(set_api_key_runtime, inputs=[key_in], outputs=[build_status])
    load_pkl_btn.click(_ui_load_cache_pkl, outputs=[build_status])
    save_pkl_btn.click(_ui_save_cache_pkl, outputs=[build_status])

    gr.Markdown("## ② 查詢（片段層）")
    with gr.Tab("關鍵字直接搜"):
        with gr.Row():
            q1 = gr.Textbox(label="輸入需求（例：『崁燈 12W 3000K CRI90 24度』）", lines=2)
            topk1 = gr.Slider(1, 15, value=6, step=1, label="Top-K")
        btn1 = gr.Button("搜尋")
        out1 = gr.Markdown()
        btn1.click(ask_basic, inputs=[q1, topk1], outputs=[out1])

    with gr.Tab("AI 理解後搜尋（LLM 先轉換）"):
        with gr.Row():
            q2 = gr.Textbox(label="例如：『適合浴室的吸頂燈』、『展示櫃用窄角高CRI』", lines=2)
            topk2 = gr.Slider(1, 15, value=6, step=1, label="Top-K")
        btn2 = gr.Button("AI 理解後搜尋")
        out2 = gr.Markdown()
        btn2.click(ask_ai, inputs=[q2, topk2], outputs=[out2])

    gr.Markdown("## ③ 以產品為中心（叢集層）")
    with gr.Tab("產品關鍵字搜尋（Cluster）"):
        with gr.Row():
            q3 = gr.Textbox(label="輸入需求（例：『LED21st DL-123 IP65 吸頂燈』）", lines=2)
            topk3 = gr.Slider(1, 15, value=6, step=1, label="Top-K")
        btn3 = gr.Button("以產品為中心搜尋")
        out3 = gr.Markdown()
        btn3.click(ask_product, inputs=[q3, topk3], outputs=[out3])

    with gr.Tab("AI 理解後產品搜尋（Cluster + 規格過濾）」"):
        with gr.Row():
            q4 = gr.Textbox(label="例如：『適合浴室的燈』、『戶外走道照明』或包含型號名稱", lines=2)
            topk4 = gr.Slider(1, 15, value=6, step=1, label="Top-K")
        btn4 = gr.Button("AI 理解後產品搜尋")
        out4 = gr.Markdown()
        btn4.click(ask_product_ai, inputs=[q4, topk4], outputs=[out4])

# 啟動時（可選）嘗試載入 PKL
def try_load_pkl_on_start():
    folder = os.path.join(os.getcwd(), "catalogs")
    if not os.path.isdir(folder):
        return f"⚠️ 找不到資料夾：{folder}"
    return load_cache_pkl(INDEX, folder)

if __name__ == "__main__":
    print(try_load_pkl_on_start())
    demo.launch()
