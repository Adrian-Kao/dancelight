# app1.py — RAG 查詢（讀取現成 PKL）+ LLM 查詢規劃 + 快捷鍵查詢
# pip install -U gradio sentence-transformers numpy python-dotenv openai regex

import os, time, json, pickle, regex as re
import numpy as np
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import time 
import random

# -----------------------------
# 0) ENV & OpenAI 初始化（新/舊 SDK 皆可）
# -----------------------------
DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(DOTENV_PATH or "", override=True)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

OPENAI = None
OPENAI_STATUS = []


def img_to_png_bytes(page, max_w=1280) -> bytes:
    """
    轉出較小的 JPEG，減少上傳大小與錯誤率。
    """
    # 先以 1.5x 渲染（畫質OK且不會爆）
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # 若太寬就縮圖
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    return buf.getvalue()

def _init_openai():
    global OPENAI, OPENAI_STATUS
    OPENAI_STATUS.clear()
    if not OPENAI_API_KEY:
        OPENAI = None
        OPENAI_STATUS.append("no_key")
        return
    try:
        from openai import OpenAI  # new sdk
        OPENAI = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_STATUS.append("new_sdk_ok")
        return
    except Exception as e:
        OPENAI_STATUS.append(f"new_sdk_fail:{type(e).__name__}")
    try:
        import openai  # legacy sdk
        openai.api_key = OPENAI_API_KEY
        OPENAI = openai
        OPENAI_STATUS.append("legacy_sdk_ok")
        return
    except Exception as e:
        OPENAI = None
        OPENAI_STATUS.append(f"legacy_sdk_fail:{type(e).__name__}")

def _chat_once(messages, model="gpt-4o-mini", max_tokens=400, temperature=0.2):
    if OPENAI is None:
        raise RuntimeError("openai_client_not_ready")
    # new sdk
    if hasattr(OPENAI, "chat") and hasattr(OPENAI.chat, "completions"):
        resp = OPENAI.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    # legacy sdk
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
    if not k.startswith("sk-"):
        return "❌ 看起來不像 OpenAI Key（需以 sk- 開頭）。"
    os.environ["OPENAI_API_KEY"] = k
    OPENAI_API_KEY = k
    _init_openai()
    return f"✅ 已套用。狀態：{', '.join(OPENAI_STATUS)}"

_init_openai()
print(f"[dotenv] path={DOTENV_PATH or '(none)'} key_loaded={bool(OPENAI_API_KEY)} status={','.join(OPENAI_STATUS)}")

# -----------------------------
# 1) 嵌入模型 & 索引類別
# -----------------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def _doc_get(d, key, default=""):
    return d.get(key, default) if isinstance(d, dict) else default

def _doc_text(d):
    return d.get("text", "") if isinstance(d, dict) else str(d)

class InMemoryIndex:
    def __init__(self):
        self.docs = []
        self.embs = None
        self.built = False

    def search(self, query: str, top_k: int = 6):
        if not self.built or self.embs is None or len(self.docs) == 0:
            return []
        q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.embs @ q_emb
        k = min(max(top_k, 1), len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        pairs = [(int(i), float(sims[i])) for i in idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def load_from_pkl(self, path: str) -> str:
        """
        高容錯 PKL 讀取：
          - InMemoryIndex 物件
          - dict{docs, embs}
          - [docs, embs] / 只有 [docs]
          - 損壞/混雜：回退成純文字行列表
        """
        import re as _re

        def _try_pickle_bytes(raw: bytes):
            try:
                return pickle.loads(raw)
            except Exception:
                # 找 pickle header (\x80\x04) 嘗試切片反序列化
                for m in _re.finditer(b"\x80\x04", raw):
                    try:
                        return pickle.loads(raw[m.start():])
                    except Exception:
                        continue
                return None

        try:
            with open(path, "rb") as f:
                raw = f.read()

            obj = _try_pickle_bytes(raw)
            if obj is None:
                # 嘗試文字來回轉碼
                try:
                    txt = raw.decode("utf-8", errors="ignore")
                    raw2 = txt.encode("latin1", errors="ignore")
                    obj = _try_pickle_bytes(raw2)
                except Exception:
                    obj = None

            if obj is None:
                # 最後回退：按行當文字
                txt = raw.decode("utf-8", errors="ignore")
                lines = [line.strip() for line in txt.splitlines() if line.strip()]
                self.docs = [{"catalog": "未知", "text": ln} for ln in lines]
                self.embs = None
                self.built = False
                return f"⚠️ 以純文字回退載入：{len(self.docs)} 行（無 embeddings）"

            docs, embs = None, None
            if hasattr(obj, "docs") and hasattr(obj, "embs"):
                docs, embs = list(obj.docs), obj.embs
            elif isinstance(obj, dict):
                docs, embs = obj.get("docs"), obj.get("embs")
            elif isinstance(obj, (list, tuple)):
                if len(obj) >= 2 and isinstance(obj[0], (list, tuple)):
                    docs, embs = list(obj[0]), obj[1]
                elif len(obj) >= 1:
                    docs, embs = list(obj), None

            if docs is None:
                return "❌ 載入失敗：未偵測到可用的 docs。"

            if embs is not None and not isinstance(embs, np.ndarray):
                try:
                    embs = np.array(embs, dtype=np.float32)
                except Exception:
                    pass

            self.docs, self.embs = docs, embs
            self.built = embs is not None and len(docs) == (embs.shape[0] if hasattr(embs, "shape") else len(embs))

            if self.built:
                dim = int(self.embs.shape[1]) if hasattr(self.embs, "shape") and self.embs.ndim == 2 else "?"
                return f"✅ 已載入快取：片段 {len(self.docs)}，維度 {dim}"
            else:
                return f"⚠️ 已讀取 docs={len(self.docs)}，但缺少 embeddings。"

        except Exception as e:
            return f"❌ 載入失敗：{type(e).__name__}: {e}"

# **全域索引實例 — 在任何函式前就建立，避免 NameError**
INDEX = InMemoryIndex()

# -----------------------------
# 2) 嵌入輔助（可選：自動補嵌入）
# -----------------------------
def embed_in_batches(texts, batch_size=128, progress=None):
    embs = []
    total = len(texts); t0 = time.time()
    for i in range(0, total, batch_size):
        if progress: progress(min(i/total, 0.98), desc=f"Embedding {i}/{total}")
        batch = texts[i:i+batch_size]
        embs.append(
            embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
        )
    if progress: progress(1.0, desc=f"Done {total}/{total} in {time.time()-t0:.1f}s")
    return np.vstack(embs) if embs else np.zeros((0, 1024), dtype=np.float32)

# -----------------------------
# 3) 基礎關鍵字查詢
# -----------------------------
def keyword_search(q: str, top_k: int = 6) -> str:
    if not INDEX.built:
        return "⚠️ 請先載入 `catalog_index.pkl`（或補嵌入）。"
    if not q.strip():
        return "請輸入關鍵字。"
    hits = INDEX.search(q.strip(), top_k=top_k)
    if not hits:
        return "❌ 沒有找到相符內容。"
    out = []
    for idx, _ in hits:
        d = INDEX.docs[idx]
        sheet = _doc_get(d, "catalog", "頁面")
        txt = _doc_text(d).replace("\n", " ")
        out.append(f"### {sheet}\n• {txt[:240]}{'…' if len(txt)>240 else ''}")
    return "\n\n".join(out)

# -----------------------------
# 4) LLM 規劃 → 檢索
# -----------------------------
_RULE_HINTS = [
    (r"浴室|衛浴|潮濕|廁所",        ["IP65","IP54","防水","防潮","防濕"]),
    (r"戶外|庭院|陽台|雨",          ["IP65","IP66","防水","耐候"]),
    (r"廚房|油煙",                  ["易清潔","防油汙","IP44"]),
    (r"走道|玄關|樓梯",             ["感應","微波感應","廣角","防眩"]),
    (r"展示櫃|展櫃|陳列",           ["窄角","15°","24°","高CRI","Ra≥90"]),
    (r"閱讀|辦公|書房",             ["4000K","5000K","防眩","無頻閃"]),
    (r"攝影|商品拍攝",              ["高CRI","Ra≥90","Ra≥95","高流明"]),
]

def _rule_plan(q: str) -> dict:
    must, nice = [], []
    for pat, kws in _RULE_HINTS:
        if re.search(pat, q):
            nice.extend(kws)
    must += re.findall(r"\b\d{2,4}K\b", q, re.I)
    must += re.findall(r"\b\d{1,3}\s?W\b", q, re.I)
    must += re.findall(r"\bIP\s?\d{2}\b", q, re.I)
    must += re.findall(r"CRI\s*(?:≥|≧)?\s*\d{2}", q, re.I)
    must = list(dict.fromkeys([m.replace(" ","") for m in must]))
    nice = list(dict.fromkeys(nice))
    return {"must_terms": must, "nice_to_have": nice, "negations": []}

def llm_plan_query(user_text: str) -> dict:
    if OPENAI is None:
        return _rule_plan(user_text)
    sys = ("你是燈具顧問。請把使用者需求轉成 JSON，含 must_terms/nice_to_have/negations 陣列；"
           "用常見規格：3000K、20W、CRI≥90、IP65、15°/24°、防水/防潮/防眩/感應/高CRI；只回 JSON。")
    usr = f"使用者需求：{user_text}\n只回 JSON，例如：{{\"must_terms\":[\"IP65\"],\"nice_to_have\":[\"高CRI\"],\"negations\":[]}}"
    try:
        txt = _chat_once([{"role":"system","content":sys},{"role":"user","content":usr}], max_tokens=200)
        data = json.loads(txt)
        if not isinstance(data, dict): raise ValueError
        for k in ("must_terms","nice_to_have","negations"):
            if k not in data or not isinstance(data[k], list): data[k] = []
        return data
    except Exception:
        return _rule_plan(user_text)

def build_search_string(plan: dict) -> str:
    must = plan.get("must_terms", [])
    nice = plan.get("nice_to_have", [])
    tokens = must + must + nice  # must 加權
    return " ".join(tokens) if tokens else ""

def search_by_planner(user_text: str, top_k: int = 6) -> str:
    if not INDEX.built:
        return "⚠️ 請先載入 `catalog_index.pkl`（或補嵌入）。"
    plan = llm_plan_query(user_text)
    q = build_search_string(plan) or user_text
    hits = INDEX.search(q, top_k=top_k)
    if not hits:
        return f"❌ 找不到結果（查詢字串：{q}）"

    spec_re = {
        "瓦數":   r"\b(\d{1,3}\s?W)\b",
        "色溫":   r"\b(\d{3,4}\s?K)\b",
        "CRI":    r"(CRI\s*(?:≥|≧)?\s*\d{2})",
        "光束角": r"(\d{1,3}\s?(?:°|度))",
        "IP":     r"\bIP\s?\d{2}\b",
        "流明":   r"\b\d{3,5}\s?lm\b",
        "特性":   r"(防水|防潮|防濕|防眩|感應|高CRI|耐候|無頻閃)"
    }
    lines = [f"**查詢理解** → must={plan.get('must_terms',[])}, nice={plan.get('nice_to_have',[])}"]
    for idx, _ in hits:
        d = INDEX.docs[idx]
        sheet = _doc_get(d, "catalog", "頁面")
        text  = _doc_text(d).replace("\n"," ")
        bullets=[]
        for k, rg in spec_re.items():
            ms = list(re.finditer(rg, text, re.I))
            if ms:
                vals = list(dict.fromkeys([m.group(1) if m.lastindex else m.group(0) for m in ms]))
                bullets.append(f"- **{k}**：{', '.join(vals[:4])}")
        if not bullets:
            bullets.append(f"- 內容：{text[:160]}{'…' if len(text)>160 else ''}")
        lines.append(f"### {sheet}\n" + "\n".join(bullets))
    return "\n\n".join(lines)

# -----------------------------
# 5) 快捷鍵查詢
# -----------------------------
_QUICK_FEATURE_MAP = {
    "亮度(lm)": ["lm","流明","高流明"],
    "色溫(K)":  ["K","3000K","4000K","5000K","6500K","暖白","白光","自然光"],
    "顯色(CRI)": ["CRI","Ra≥80","Ra≥90","高CRI","CRI≥90"],
    "光束角(°)": ["°","度","15°","24°","36°","45°"],
    "防護(IP)": ["IP65","IP66","IP54","防水","防潮","耐候"],
    "功率(W)":  ["W","瓦","12W","15W","20W","30W","50W"],
}

def quick_feature_search(name_or_model: str, feature: str, top_k: int = 6) -> str:
    if not INDEX.built:
        return "⚠️ 請先載入 `catalog_index.pkl`（或補嵌入）。"
    if not name_or_model.strip():
        return "請先輸入型號或品名關鍵字。"
    seeds = _QUICK_FEATURE_MAP.get(feature, [])
    q = " ".join([name_or_model.strip()] + seeds)
    hits = INDEX.search(q, top_k=top_k)
    if not hits:
        return f"找不到：{name_or_model} 的「{feature}」相關資訊。"
    out=[]
    for idx,_ in hits:
        d=INDEX.docs[idx]
        sheet=_doc_get(d,"catalog","頁面")
        txt=_doc_text(d).replace("\n"," ")
        for kw in seeds+[feature, name_or_model]:
            k=kw.strip()
            if not k: continue
            try: txt = re.sub(re.escape(k), f"**{k}**", txt, flags=re.I)
            except re.error: pass
        out.append(f"### {sheet}\n• {txt[:240]}{'…' if len(txt)>240 else ''}")
    return "\n\n".join(out)

# -----------------------------
# 6) 載入/診斷/補嵌入
# -----------------------------
CACHE_PATH = "catalog_index.pkl"

def load_cache_btn(auto_embed: bool=False, progress=gr.Progress()):
    """嘗試載入；若缺 embs 且勾選 auto_embed → 現場補嵌入並寫 merged 檔"""
    global INDEX
    if not os.path.exists(CACHE_PATH):
        return f"⚠️ 找不到 `{CACHE_PATH}`，請確認檔案位置。"
    msg = INDEX.load_from_pkl(CACHE_PATH)
    if msg.startswith("✅"):
        return msg
    if ("缺少 embeddings" in msg or "無 embeddings" in msg) and auto_embed:
        texts = [_doc_text(d) for d in INDEX.docs]
        if not any(texts):
            return "❌ docs 內沒有可嵌入的文字。"
        INDEX.embs = embed_in_batches(texts, batch_size=128, progress=progress)
        INDEX.built = True
        merged = "catalog_index_merged.pkl"
        with open(merged, "wb") as f:
            pickle.dump({"docs": INDEX.docs, "embs": INDEX.embs}, f)
        return f"✅ 已自動補嵌入：片段 {len(texts)}，維度 {INDEX.embs.shape[1]}\n💾 已寫出 `{merged}`（下次可直接載入）"
    return msg

def peek_pkl(path=CACHE_PATH):
    if not os.path.exists(path):
        return f"❌ 檔案不存在：{os.path.abspath(path)}"
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        info = {"type": str(type(obj))}
        if hasattr(obj, "docs") and hasattr(obj, "embs"):
            info["summary"] = f"InMemoryIndex-like: docs={len(obj.docs)}, embs={getattr(obj.embs,'shape',None)}"
        elif isinstance(obj, dict):
            info["summary"] = f"dict keys={list(obj.keys())}"
        elif isinstance(obj, (list, tuple)):
            info["summary"] = f"sequence len={len(obj)}; elem0={type(obj[0]) if obj else None}"
        else:
            info["summary"] = "unknown layout"
        return "🔎 PKL 探測：\n```\n" + json.dumps(info, ensure_ascii=False, indent=2) + "\n```"
    except Exception as e:
        return f"❌ 解析失敗：{type(e).__name__}: {e}"

def diagnostics():
    lines = []
    lines.append(f"📦 檔案：{os.path.abspath(CACHE_PATH)} 存在={os.path.exists(CACHE_PATH)}")
    if INDEX and getattr(INDEX, 'built', False) and INDEX.embs is not None:
        lines.append(f"✅ 索引可用：片數={len(INDEX.docs)}, 維度={INDEX.embs.shape[1]}")
    else:
        lines.append("⚠️ 索引尚未可用（未載入或缺 embeddings）")
    lines.append(f"🧠 OpenAI 狀態：{', '.join(OPENAI_STATUS) or 'unknown'}")
    lines.append(f"🔡 嵌入模型：{EMBED_MODEL_NAME}")
    return "\n".join(lines)

# -----------------------------
# 7) Gradio 介面
# -----------------------------
with gr.Blocks(title="Lighting Catalog — RAG（PKL）") as demo:
    gr.Markdown("## 💡 Lighting Catalog — RAG（載入 PKL）\n"
                "- 先載入 `catalog_index.pkl`；若只有文字無向量，可勾選「自動補嵌入」。\n"
                "- 提供：基礎關鍵字搜尋 / AI 理解後搜尋 / 快捷鍵查詢。\n")

    with gr.Row():
        auto_embed_chk = gr.Checkbox(label="若 PKL 無向量，自動補嵌入並輸出 merged 檔", value=False)
        load_btn = gr.Button("📦 載入 catalog_index.pkl", scale=2)
        peek_btn = gr.Button("🔍 檔案結構探測", scale=1)
        diag_btn = gr.Button("🧪 診斷", scale=1)
        key_box = gr.Textbox(label="（可選）臨時設定 OpenAI API Key：sk-...", type="password")
        key_btn = gr.Button("套用Key", scale=1)

    status = gr.Markdown("尚未載入索引")
    load_btn.click(fn=load_cache_btn, inputs=[auto_embed_chk], outputs=[status])
    peek_btn.click(fn=peek_pkl, outputs=[status])
    diag_btn.click(fn=diagnostics, outputs=[status])
    key_btn.click(fn=set_api_key_runtime, inputs=[key_box], outputs=[status])

    gr.Markdown("### 🔎 基礎關鍵字搜尋")
    with gr.Row():
        q_basic = gr.Textbox(label="關鍵字/型號（如：30W 3000K CRI90 / MX9）", lines=2)
        topk_basic = gr.Slider(1, 20, value=6, step=1, label="Top-K")
    btn_basic = gr.Button("搜尋")
    out_basic = gr.Markdown()
    btn_basic.click(fn=keyword_search, inputs=[q_basic, topk_basic], outputs=[out_basic])

    gr.Markdown("### 🤖 AI 理解後搜尋（LLM 先把需求轉成檢索條件）")
    with gr.Row():
        q_ai = gr.Textbox(label="例如：『適合浴室的吸頂燈』、『展示櫃用窄角高CRI』", lines=2)
        topk_ai = gr.Slider(1, 20, value=6, step=1, label="Top-K")
    btn_ai = gr.Button("AI 理解後搜尋")
    out_ai = gr.Markdown()
    btn_ai.click(fn=search_by_planner, inputs=[q_ai, topk_ai], outputs=[out_ai])

    gr.Markdown("### ⚡ 快捷鍵查詢（型號/品名 → 一鍵看特徵）")
    with gr.Row():
        quick_name = gr.Textbox(label="型號/品名（如：MX9 / 舞光軌道燈 / 30W 神盾）")
        quick_feat = gr.Radio(choices=list(_QUICK_FEATURE_MAP.keys()), value="防護(IP)", label="特徵")
        quick_topk = gr.Slider(1, 20, value=6, step=1, label="Top-K")
    btn_quick = gr.Button("查這個特徵")
    out_quick = gr.Markdown()
    btn_quick.click(fn=quick_feature_search, inputs=[quick_name, quick_feat, quick_topk], outputs=[out_quick])

if __name__ == "__main__":
    demo.launch()
