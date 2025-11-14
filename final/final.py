#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
product_filter_ui.py
Gradio 版 — 讀取 merged_products.json，提供「查詢」與「篩選」介面（中文介面）
"""

import os, json, re, gradio as gr

DATA_FILE = "merged_products.json"

# ============= 讀取 JSON =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "merged_products.json")
def load_products():
    if not os.path.exists(DATA_FILE):
        return [], f"❌ 找不到 {DATA_FILE}，請先確認檔案存在。"
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], f"❌ 檔案格式錯誤：應為陣列。"
        return data, f"✅ 已載入 {len(data)} 筆資料。"
    except Exception as e:
        return [], f"❌ 載入失敗：{e}"

products, load_msg = load_products()

# ============= 查詢功能 =============
def search_product(keyword: str):
    if not keyword or not keyword.strip():
        return "⚠️ 請輸入關鍵字。"
    if not products:
        return "⚠️ 尚未載入產品資料。"

    q = keyword.strip().lower()
    results = []
    for p in products:
        model = str(p.get("model", "")).lower()
        if q in model:
            results.append(p)
    if not results:
        return f"❌ 找不到與「{keyword}」相關的產品。"

    lines = [f"### 🔎 查詢結果：{len(results)} 筆\n"]
    for it in results[:30]:
        lines.append(
            f"- **{it.get('model','未命名')}** | "
            f"功率：{it.get('watt','?')}W | 色溫：{it.get('cct','?')}K | "
            f"光束角：{it.get('beam','?')}° | 光通量：{it.get('lumen','?')}lm | "
            f"價格：{it.get('price','?')} 元"
        )
    return "\n".join(lines)

# ============= 篩選功能 =============
def filter_products(
    keyword,
    watt_lo, watt_hi,
    cct_lo, cct_hi,
    beam_lo, beam_hi,
    lumen_lo, lumen_hi,
    price_lo, price_hi,
    topk
):
    if not products:
        return "⚠️ 尚未載入產品資料。"

    base = products
    if keyword and keyword.strip():
        q = keyword.strip().lower()
        base = [p for p in products if q in str(p.get("model","")).lower()]
        if not base:
            return f"❌ 找不到與「{keyword}」相關的產品。"

    def num(v):
        try: return float(v)
        except: return 0

    result = []
    for p in base:
        w = num(p.get("watt",0))
        c = num(p.get("cct",0))
        b = num(p.get("beam",0))
        l = num(p.get("lumen",0))
        pr= num(p.get("price",0))
        if not (watt_lo<=w<=watt_hi): continue
        if not (cct_lo <=c<=cct_hi): continue
        if not (beam_lo<=b<=beam_hi): continue
        if not (lumen_lo<=l<=lumen_hi): continue
        if not (price_lo<=pr<=price_hi): continue
        result.append(p)

    if not result:
        return "❌ 沒有符合條件的產品。"

    lines = [f"### 篩選結果：共 {len(result)} 筆（顯示前 {int(topk)} 筆）\n"]
    for it in result[:int(topk)]:
        lines.append(
            f"- **{it.get('model','未命名')}** | "
            f"功率：{it.get('watt','?')}W | 色溫：{it.get('cct','?')}K | "
            f"光束角：{it.get('beam','?')}° | 光通量：{it.get('lumen','?')}lm | "
            f"價格：{it.get('price','?')} 元"
        )
    return "\n".join(lines)

# ============= Gradio UI =============
with gr.Blocks(title="燈具規格查詢與篩選") as demo:
    gr.Markdown("# 💡 燈具規格查詢與篩選系統")
    gr.Markdown(load_msg)

    # 查詢區
    gr.Markdown("## 🔍 查詢產品")
    with gr.Row():
        query_input = gr.Textbox(label="輸入型號或關鍵字", placeholder="例如：T5、D-FXTR7N、軌道燈…", scale=4)
        btn_search = gr.Button("查詢", variant="primary", scale=1)
    search_output = gr.Markdown()
    btn_search.click(search_product, inputs=[query_input], outputs=[search_output])

    # 篩選區
    gr.Markdown("## 🧾 屬性篩選")
    series_input = gr.Textbox(label="系列關鍵字（可留空）", placeholder="例如：T5、D-T5BA1、OD 系列等")

    with gr.Row():
        watt_lo = gr.Slider(0,200,0,step=1,label="功率最小 (W)")
        watt_hi = gr.Slider(0,200,200,step=1,label="功率最大 (W)")
    with gr.Row():
        cct_lo = gr.Slider(2000,7000,2700,step=50,label="色溫最小 (K)")
        cct_hi = gr.Slider(2000,7000,6500,step=50,label="色溫最大 (K)")
    with gr.Row():
        beam_lo = gr.Slider(0,120,0,step=1,label="光束角最小 (°)")
        beam_hi = gr.Slider(0,120,120,step=1,label="光束角最大 (°)")
    with gr.Row():
        lumen_lo = gr.Slider(0,15000,0,step=10,label="光通量最小 (lm)")
        lumen_hi = gr.Slider(0,15000,15000,step=10,label="光通量最大 (lm)")
    with gr.Row():
        price_lo = gr.Slider(0,200000,0,step=100,label="價格最小")
        price_hi = gr.Slider(0,200000,200000,step=100,label="價格最大")
    topk = gr.Slider(1,50,20,step=1,label="最多顯示筆數")

    btn_filter = gr.Button("開始篩選", variant="primary")
    filter_output = gr.Markdown()
    btn_filter.click(
        filter_products,
        inputs=[series_input,watt_lo,watt_hi,cct_lo,cct_hi,beam_lo,beam_hi,lumen_lo,lumen_hi,price_lo,price_hi,topk],
        outputs=[filter_output]
    )

if __name__ == "__main__":
    demo.launch()
