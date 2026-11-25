# 類神經網路作業三｜Hopfield 與 SOM

**第一步**　先跑 `python -m pytest tests/test_data.py -k patterns -q`，再到 `src/hw3/data.py` 實作 `load_patterns`。這一題會告訴你解析圖案時哪些空格必須留下來。之後依 [任務清單](TASKS.md) 選 Hopfield 或 SOM 接著做。

[課程說明](../類神經網路作業三說明.pdf) 第 1 頁允許二選一。本專案依需求涵蓋兩者。第 2 頁的截止時間為 **2026/12/09（三）23:59**。目前交付階段是測試先行，函式簽名與資料結構已建好，演算法與視窗保留給實作者。函式中的 `NotImplementedError` 會指出對應任務。

## 架構與資料流

```text
hw3/
├── main.py                 啟動 Tk，將 data 路徑交給視窗
├── src/hw3/
│   ├── __init__.py
│   ├── data.py             讀圖案、拆 SOM 特徵與標籤
│   ├── hopfield.py         train、recall、add_noise
│   ├── som.py              bmu、step、train
│   ├── workflows.py        檔案 → 模型 → RecallCase / SomRun
│   └── gui.py              Hopfield 與 SOM 分頁、畫布及操作事件
├── data/
│   ├── hopfield/           Basic 與 Bonus 各一對原始 txt
│   └── som/                課程提供的每個 SOM 資料集
├── tests/
│   ├── conftest.py         可手算的小型資料、驗收參數
│   ├── test_data.py        檔案解析
│   ├── test_hopfield.py    權重、回想、雜訊
│   ├── test_som.py         得勝格點、鄰域更新與迭代
│   ├── test_workflows.py   資料配對、模型輸出、錯誤訊息
│   ├── test_gui.py         以 Tk 元件操作按鈕並讀回畫布
│   ├── test_course_data.py 正式附件驗收
│   └── test_submission.py  繳交壓縮檔驗收
├── requirements-dev.txt
├── pyproject.toml          pytest 設定
├── README.md
└── TASKS.md
```

GUI 將路徑與使用者參數交給 `workflows`。應用層呼叫 `data` 解析資料，再呼叫模型。畫布直接使用應用層的回傳值。實作採 Python 3.10+、Tkinter 與 NumPy；矩陣運算以 NumPy 進行，兩個網路的學習規則自行撰寫。封裝目標為可雙擊的 Windows 執行檔，報告採 PDF，壓縮格式採 ZIP。

## 第一次執行

在本目錄開 PowerShell，使用已安裝 Tkinter 的 Python。下例使用虛擬環境內的直譯器，之後的 `python -m pytest` 指令都可使用同一路徑。

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest --collect-only -q
.\.venv\Scripts\python -m pytest tests/test_data.py -k patterns -q
```

首行使用本機已確認的 Windows Python 安裝路徑。其他電腦可改用自己的 Python 執行 `python -m venv .venv`。

| 驗收時機 | 從 hw3 目錄執行 |
| --- | --- |
| 日常開發，執行合成資料與正式附件測試 | `python -m pytest -q -rs` |
| 先做基本功能，不開視窗 | `python -m pytest -m "not bonus and not gui and not course and not release"` |
| 單獨測 Hopfield 工作流程 | `python -m pytest tests/test_workflows.py -k "hopfield or recall"` |
| 單獨測 SOM 演算法 | `python -m pytest tests/test_som.py` |
| 桌面 GUI，Tk 啟動失敗也要回報 | `python -m pytest -m gui --require-gui` |
| 確認 28 個正式檔案就位（可在實作前執行） | `python -m pytest -q -k course_files_present` |
| 逐一驗收正式資料並記錄結果 | `python -m pytest -m course --junitxml=course-results.xml` |
| 封裝後的完整驗收 | `python -m pytest --require-gui --course-data data --submission "dist/學號_姓名_作業三.zip"` |

最後一條請代入真正的學號與姓名。正式資料預設從本專案的 `data/` 讀取，也可用 `--course-data` 指定另一份解壓目錄。繳交包測試在提供 `--submission` 後執行。沒有圖形環境時可先跑模型測試，交付前須在桌面以 `--require-gui` 再驗。開發時使用 `python main.py` 啟動已實作的 GUI。

## 課程資料

| 來源 | 放置位置 | 已核對的內容 |
| --- | --- | --- |
| `Hopfield_dataset.7z` 的 Basic | `data/hopfield/Basic_{Training,Testing}.txt` | 3 組配對，每圖 12 列 × 9 欄。三張測試圖與對應訓練圖分別相差 15、22、32 個像素。 |
| 同一個 7z 的 Bonus | `data/hopfield/Bonus_{Training,Testing}.txt` | 15 組配對，每圖 10 列 × 10 欄。有些圖含整列空格，這一列也是圖的一部分。 |
| `SOM_dataset.zip` 的 `DataSet/` | `data/som/` | 24 個文字檔，共 8,089 筆。每筆有 2 至 25 個特徵，最右欄是預期分群。原始檔名與內容都已保留。 |

SOM 可從 `123.txt`（4 筆、2 維）開始，再用 `IRIS.TXT`（150 筆、4 維）驗證大寫副檔名與小數標籤。`Number.txt` 有 25 個特徵，`4satellite-6.txt` 有 3,772 筆。全部檔名、列數與特徵數列在 [正式資料驗收測試](tests/test_course_data.py) 的 `SOM_DATASETS`。每個檔名都有獨立的 pytest 結果。

## 實作契約

下列讀檔規則已用正式附件核對。模型的更新順序、零門檻與高斯鄰域是本專案的實作選擇。

| 模組 | 輸入、輸出與可觀察行為 |
| --- | --- |
| Hopfield 讀檔 | 原檔為 ASCII，使用 CRLF 換行，最後一列沒有換行符號。字元 `1` 映射為 +1，空格映射為 −1，完全空行分圖。只移除換行符號，保留行首、行尾與全空格的像素列。依檔內順序讀取，輸出形狀為 (圖數, 高, 寬)。 |
| Hopfield 訓練與回想 | 令 X 每列為一張展平圖，N 為每圖像素數。W = XᵀX / N，對角清零。本版選用雙極編碼與零門檻，依像素索引就地非同步更新，淨輸入為零時保留原值。一整輪沒有翻轉即可停止，另設 max_sweeps 上限。輸入圖保持可供比對的原樣。 |
| Hopfield 工作流程 | 只用訓練檔建 W，每張測試圖各自回想；按索引回傳目標圖、測試圖、回想圖。加雜訊時從訓練圖複本翻轉指定數量的不重複像素，以相同種子重現結果。 |
| SOM 讀檔 | 正式附件為無表頭的 ASCII 文字檔，每列一筆，以空白分欄。保留最右欄標籤的原始字串（例如 `-1`、`1.0`、`1.00`），其餘欄位全部轉為浮點特徵。GUI 清單須涵蓋 `.txt` 與 `.TXT`。 |
| SOM 更新 | 權重形狀為 (格點列, 格點欄, 特徵數)。BMU 是特徵空間中歐氏距離最小的格點。高斯鄰域係數 h = exp(−d² / (2r²))，d 是格點到 BMU 的網格距離。一次更新為 w ← w + rate × h × (x − w)。 |
| SOM 迭代與呈現 | 一次迭代處理一筆，依序輪取樣本。首步使用傳入學習率和半徑；後續可固定或逐步衰減並記錄設定。初始權重由 seed 決定，歷程保存初始畫面與每步獨立快照。最後為每筆樣本回傳得勝格點。標籤供顯示與討論。 |
| GUI | `build_app(root, data_dir)` 回傳兩個分頁控制物件。Hopfield 分頁可選資料、執行、下一張及加雜訊；SOM 分頁可選檔、填迭代次數、執行、檢視下一個拓樸畫面。Tk 元件屬性與 Canvas tag 見 `test_gui.py`，可自行設計視窗配置。高維 SOM 可用前兩個特徵顯示投影，計算仍使用全部特徵。 |

## 第一版錯誤訊息

統一使用 `[代碼] 使用者可理解的原因。`。應用層拋出 `ValueError`，GUI 接住後顯示同一段文字。第一版固定兩種情況。

| 觸發條件 | 訊息範例 |
| --- | --- |
| 訓練圖與測試圖數量不一致 | `[DATA_PAIR_COUNT] 訓練圖有 2 張，測試圖有 1 張。`（數量代入實際值） |
| 迭代次數未填正整數 | `[SOM_ITERATIONS] 迭代次數必須為正整數。` |

## 判讀結果

合成圖的預期回想結果已知，可逐像素驗證。正式資料測試另外記錄每組完整回想張數與總張數，可在 JUnit XML 的 `exact_recalls` 查看。SOM 格點沒有固定類別編號，測試以手算更新值、得勝格點及標籤隔離驗證模型。正式成績相關的實驗結果、圖形可讀性與乾淨電腦雙擊啟動，依任務清單留下人工驗收紀錄。
