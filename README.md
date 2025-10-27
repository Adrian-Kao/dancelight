
建議專案結構

```
LightingCatalog/
│
├── app.py                  ← 主程式入口
├── requirements.txt        ← 所需套件清單
├── catalogs/               ← 放 PDF 型錄的資料夾
└── catalog_index.pkl       ← (選擇性) OCR + 向量索引快取檔
```

---

## ⚙️ 1. 建立虛擬環境（venv）

### Windows PowerShell

```bash
# 建立虛擬環境
python -m venv venv

# 啟用虛擬環境
.\venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 若不確定 Python 版本，請使用 Python 3.9 以上。

---

## 📦 2. 安裝套件

執行：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

若尚未建立 `requirements.txt`，可使用以下內容：

```txt
gradio
pytesseract
pdfplumber
PyMuPDF
sentence-transformers
torch
numpy
pandas
scikit-learn
Pillow
```

---

## 🔠 3. 安裝 Tesseract OCR

### Windows

1. 前往 [Tesseract 官方下載頁](https://github.com/UB-Mannheim/tesseract/wiki)
2. 下載並安裝最新版（例如 `tesseract-ocr-w64-setup.exe`）
3. 安裝後將執行檔加入系統環境變數，例如：

   ```
   C:\Program Files\Tesseract-OCR
   ```
4. 驗證：

   ```bash
   tesseract --version
   ```

### macOS

```bash
brew install tesseract
```

### Linux (Ubuntu)

```bash
sudo apt install tesseract-ocr
```

> 若需要支援中文 OCR，安裝語言包：
>
> ```bash
> sudo apt install tesseract-ocr-chi-tra
> ```
>
> 並確保在程式中設定：
>
> ```python
> pytesseract.image_to_string(img, lang="chi_tra+eng")
> ```

---

## 📚 4. 放置 PDF 型錄

將要查詢的型錄放進專案資料夾下的：

```
catalogs/
```

例如：

```
catalogs/catalogA.pdf
catalogs/catalogB.pdf
```

---

## 🚀 5. 執行專案

```bash
python app.py
```

執行後，終端機會顯示：

```
Running on local URL:  http://127.0.0.1:7860
```

開啟瀏覽器進入此網址即可使用。

---

## ⚡ 6. 快取機制說明

第一次執行時：

* 系統會自動執行 OCR、切片、向量化並建立索引。
* 完成後會生成 `catalog_index.pkl` 快取檔。

之後再次執行 `python app.py` 時，
系統會自動載入快取，不需重新分析。

若你新增或修改型錄內容，請刪除該快取檔重新建立。

---

## 🧠 模型說明

| 模型                       | 功能           | 說明                |
| ------------------------ | ------------ | ----------------- |
| `pytesseract`            | OCR          | 從 PDF 圖像擷取文字（含中文） |
| `BAAI/bge-m3`            | Embedding 模型 | 將文字轉為語意向量，用於檢索    |
| `BAAI/bge-reranker-base` | Reranker 模型  | 精準重排序，提高語意匹配準確度   |

---

## 🧩 可選項目

若需手動重建索引，可在 Gradio 介面勾選「強制重新掃描」再執行。

---

## 🧰 疑難排解

| 問題                            | 解法                           |
| ----------------------------- | ---------------------------- |
| `tesseract is not recognized` | 將 Tesseract 安裝目錄加入 PATH      |
| `RuntimeError: No CUDA GPUs`  | 改用 CPU 模式（自動 fallback）       |
| 無法載入快取                        | 刪除 `catalog_index.pkl` 並重新建立 |

---


---

要我幫你生成對應的 `requirements.txt` 檔案內容（根據你 app.py 的實際 import），
我可以直接列出一份精準版本讓你附在專案裡。要嗎？
