# SSIN-DataMining-Final

Optimize the SSIN by adjusting its architecture and parameters to improve performance.  
For academic research use only.

---

## 1. 專案簡介

本專題以原始 **SSIN（Spatio-Temporal Sequence Imputation Network）** 為基礎，  
針對模型架構與超參數（特別是 Transformer Encoder 的層數與 multi-head 數量、embedding 維度）進行調整，  
並在 **PEMS-BW**（程式中以 `bw` 表示）資料集上完成訓練、評估與 ensemble，  
最終在課程指定的評分指標上達到約 **0.9665** 的表現。

本專案主要修改與使用的程式檔案包含：

- `main_train.py`：**主訓練腳本（本專題有修改 SSIN 部分 source code）**
- `show_scores.py`：對單一實驗的 `test_ret.csv` 計算評分指標
- `eval_all_checkpoints.py`：自動對某實驗底下所有 checkpoint 做 testing，產生多個 `test_ret.csv`
- `ensemble_csv.py`：對多個 `test_ret.csv` 做 ensemble，搜尋較佳的組合

---

## 2. 環境與專案取得方式

### 2.1 開發／執行環境（建議）

- 作業系統：Windows / Linux / macOS 皆可
- Python：3.x（建議 3.8 以上）
- 深度學習框架：
  - `PyTorch`（請依自身 CUDA / 顯卡環境自行安裝合適版本）
- 其他常用套件：
  - `numpy`
  - `pandas`
  - 以及程式中有 import 到的套件

> 套件安裝請依照實際錯誤訊息補齊，例如：
> ```bash
> pip install numpy pandas
> # PyTorch 請依照官方說明安裝，需支援 CUDA
> ```

### 2.2 專案取得

請直接 **clone 此 GitHub 專案**：

```bash
git clone <本專案的 GitHub 連結>
cd SSIN    # 以下所有指令預設在專案根目錄下執行
