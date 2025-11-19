# SSIN-DataMining-Final

Optimize the SSIN by adjusting its architecture and parameters to improve performance.  
For academic research use only.

---
### Data Mining Final Project: SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation 之模型架構調整
#### 組員: M11407301 黃奕翔、M11407321 方柏又、M11407322 陳弘典、M11407302 陳美雯、M11407329 藍煒翔
---

## 1. 專案簡介

本專題以原始 **SSIN（Spatio-Temporal Sequence Imputation Network）** 為基礎，  
針對模型架構與超參數（特別是Transformer Encoder的架構、層數、multi-head數量、embedding維度）進行調整，  
並在 **PEMS-BW**（程式中以 `bw` 表示）資料集上完成訓練、評估與 ensemble，  
最後在RMSE的指標上達到 **0.9665** 的表現，比原始論文的成績進步約2%，NSE的表現則為0.5362，進步3.96%。

本專案主要**修改**與使用的程式檔案包含：

- `main_train.py`：**主程式**
- `show_scores.py`：對單一實驗的 `test_ret.csv` 計算評分指標
- `eval_all_checkpoints.py`：自動對某實驗底下所有 checkpoint 做 testing，產生多個 `test_ret.csv`
- `ensemble_csv.py`：對多個 `test_ret.csv` 做 ensemble，搜尋較佳的組合
- `.\networks\Models.py`: 修改transformer架構與optimizer方法...等
---

## 2. 環境與專案取得方式

### 2.1 開發環境

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
> # PyTorch 請依照官方說明安裝，本專題需支援CUDA
> ```

### 2.2 專案下載
Reference: https://dl.acm.org/doi/10.1145/3589321

Source Code: https://github.com/jlidw/SSIN

Dataset & Output: https://drive.google.com/file/d/1jNp8HtFMmmA3pRHfJOYVkLPIivENroEG/view?usp=sharing

施作本專題請直接 **clone 此 GitHub 專案與 Dataset & Output壓縮檔** (因為github本身有檔案大小限制，沒辦法提供完整checkpoint與dataset)

1. 先解壓縮本Github專案，你會看到SSIN資料夾
2. 將Dataset & Output 解壓縮放到SSIN目錄下(讓data與output資料夾在SSIN中)
3. 即可開始依步驟使用。

或是 **直接下載google drive連結** https://drive.google.com/file/d/17G49WsKw_zGoLSZ7OpZyl1Hn4EpJScgK/view?usp=sharing 本連結可以直接下載使用，無須其他操作。

---

## 3. 資料準備

本專題的實驗以 BW 資料集 為主（對應 --dataset bw）。

請確認資料夾結構如下（相對於專案根目錄）：

+ ./data/
  + BW_132_data/
    + pkl_data/
      + train/
        + 2012-2014_data.pkl
      + test/
        + 2012-2014_data.pkl


main_train.py 會根據 --dataset 自動設定路徑（以 bw 為例）：

訓練資料：
`./data/BW_132_data/pkl_data/train/2012-2014_data.pkl`

測試資料：
`./data/BW_132_data/pkl_data/test/2012-2014_data.pkl`

原始論文需自行下載dataset，本git專案已經安裝完畢，無須額外下載。

---

## 4. 目錄與輸出

訓練完成後，輸出會放在：

`./output/BW_output/main/<實驗資料夾>/`

因此路徑可能會長的像：

`./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_<時間戳>/`

其中L4、H8這些關係到我們training的參數怎麼下，故請記得參數設定，方便train了多次模型後仍能找到。

一個典型的實驗資料夾內部結構會類似：

+ ./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057/
  + args_settings.txt                     # 參數設定記錄
  + train/
    + checkpoints_path/         # 每 10 個 epoch 儲存一個 checkpoint
      + checkpoint_10epoch.pth
      + checkpoint_20epoch.pth
      + ...
  + test/
    + test_ret.csv              # 全部 epoch 跑完後的最終 testing 結果


>注意：
>每次執行 main_train.py 會依照當下時間產生新的 <時間戳> 實驗資料夾，不會覆蓋舊結果。
>若要做 ensemble 或重新計算分數，需要知道欲使用的實驗資料夾路徑。

---

## 5. 實驗設定總覽（本專題最終結果）

本專題製作會採用兩組實驗設定來延伸，而參數皆以dataset=bw、model_type=SpaFormer為主，
架構已經在.\networks\models.py中修改過，2組實驗主要差異在於Transformer Encoder的層數與head數，以及embedding維度：

實驗 A
- n_layers = 4
- n_head = 8
- d_model = 32

實驗 B
- n_layers = 5
- n_head = 12
- d_model = 64

以下會示範如何在終端機下正確給參數，重現這兩組設定。

---

## 6. 實驗流程

完整操作流程依照下列順序：

1. 準備環境與資料（見上方第 2、3 節）
2. 執行單一模型訓練：使用main_train.py跑「實驗 A」與「實驗 B」**會生成2組實驗的output資料夾，步驟3開始將會用到該路徑!**
3. 使用show_scores.py對單一實驗做評分， **記得更改路徑!**
4. 使用eval_all_checkpoints.py對某一實驗底下所有checkpoint做testing **記得更改路徑!**
5. 使用ensemble_csv.py對多個checkpoint的結果做 ensemble，取得最終約 0.9665 的成績 **記得更改路徑!**

直接使用我已保留的實驗輸出做評分與 ensemble，以完全重現報告數值

以下逐步說明。

---

## 7. 單一模型訓練：main_train.py
### 7.1 main_train.py 參數說明

在程式開頭中，可看到主要參數：

資料相關

- --dataset：bw / hk / bay（本專題使用 bw）


模型相關

- --model_type：預設 "SpaFormer"

- --n_layers：Transformer encoder layer數

- --n_head：multi-head attention的head數

- --d_model：embedding維度（本專題重點調整參數之一）

訓練相關

- --epochs：訓練總epoch數（若有調整，請依實際狀況）

<img width="541" height="21" alt="image" src="https://github.com/user-attachments/assets/7551eab8-fc0e-4343-909b-d620657aa59c" />


### 7.2 實驗 A：n_layers=4, n_head=8, d_model=32, epochs=150 (使用GTX 1080 Ti 大約費時2小時)

請先切到專案根目錄：

- cd SSIN

執行指令範例：

`python main_train.py --dataset bw --n_layers 4 --n_head 8 --d_model 32 --epochs: 150`

訓練結束後，會在類似下列路徑產生實驗資料夾(資料夾名稱一定不同，因跟隨時間戳)：

`./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057/`


（實際時間戳會依執行當下時間而異）

### 7.3 實驗 B：n_layers=5, n_head=12, d_model=64, epochs=300  (使用GTX 1080 Ti 大約費時10小時)

同樣在專案根目錄下執行：

`python main_train.py --dataset bw  --n_layers 5 --n_head 12 --d_model 64`

訓練完成後，會產生類似路徑(資料夾名稱一定不同，因跟隨時間戳)：

`./output/BW_output/main/D16_L5_H12_TrainOn2012-2014_TestOn2012-2014-expB_1114-134857/`

---

## 8. 計算評分：show_scores.py

<img width="795" height="70" alt="image" src="https://github.com/user-attachments/assets/47e78ffa-acbd-4e3d-ae9e-8ded58f8e485" />

當某一實驗訓練完成後，會在該實驗資料夾下的 test 子資料夾產生：

`<實驗資料夾>/test/test_ret.csv`

接著即可使用 show_scores.py 計算該 test_ret.csv 的評分指標。

### 8.1 執行方式

在專案根目錄下執行：

`python show_scores.py`

此腳本會依內部實作，讀取對應的test_ret.csv，並在終端機中輸出各項指標（如MAE、RMSE等）。

---

## 9. 檢驗所有 checkpoints：eval_all_checkpoints.py

由於我們不預先知道哪一個 epoch 的表現最好，
因此撰寫了 eval_all_checkpoints.py 來自動對 某一實驗資料夾下的所有 checkpoint 逐一做 testing。

### 9.1 使用時機

先完成「實驗 A」的訓練（或「實驗 B」）。

在 <實驗資料夾>/train/checkpoints_path/ 中，通常每 10 個 epoch 會儲存一個 checkpoint，例如：

+ .\train\checkpoints_path
  + ckpt_epoch_10.pth
  + ckpt_epoch_20.pth
  + ...

執行 eval_all_checkpoints.py，讓它對這些 checkpoint 逐一進行 testing，並在對應的路徑產生多個 test_ret.csv。

### 9.2 注意事項

程式中在 eval_all_checkpoints.py 裡面的路徑是寫死的，請先打開檔案，找到路徑設定，例如（示意）：

`EXP_DIR = "./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057"`

請將此路徑改成你要檢驗的實驗資料夾。

修改完成後，在專案根目錄執行：

`python eval_all_checkpoints.py`

執行完後，該實驗底下會對應產生多個 test_ret.csv（依照腳本設計，可能按 epoch 或 checkpoint 命名），
之後可再搭配 show_scores.py 或 ensemble_csv.py 做分析。

重要：

做完實驗A與實驗B後，請先將 eval_all_checkpoints.py 內的路徑設定為「實驗A」的資料夾跑一次再將「實驗B」的資料夾跑一次。

如此兩組實驗的所有 checkpoint 都會產生對應的 test_ret.csv，方便後續 ensemble。

<img width="1177" height="413" alt="image" src="https://github.com/user-attachments/assets/88740f6d-3a35-422b-9d00-d69c841270ac" />


---

## 10. Ensemble：ensemble_csv.py

在取得多個 checkpoint 對應的 test_ret.csv 後，
可以透過 ensemble_csv.py 對不同組合的 checkpoint 結果進行 ensemble，
尋找更佳的整體表現。最終本專題透過該流程得到約 0.9665 的成績。

### 10.1 使用流程

確定「實驗 A」與「實驗 B」底下的所有 checkpoint 都已經由 eval_all_checkpoints.py 產生對應的 test_ret.csv。
- python ensemble_csv.py
程式會逐一測試各種checkpoint組合進行ensemble(利用Greedy forward selection與Hill climbing algorithm)，輸出對應分數，
並可找到最終表現約 0.9665 的組合。

由於實際 ensemble 策略詳細實作寫在 ensemble_csv.py 內，
若需了解演算法細節（例如是否為簡單平均、如何選擇組合等），
請直接參考該檔案中的程式碼與註解。

<img width="1158" height="653" alt="image" src="https://github.com/user-attachments/assets/09b61f75-eec4-42a0-bf35-0cc13adf620f" />

<img width="377" height="557" alt="image" src="https://github.com/user-attachments/assets/e47d30ae-f820-4e66-97c4-85827f163e13" />

<img width="973" height="666" alt="image" src="https://github.com/user-attachments/assets/72acead0-4e65-4d94-82b0-150fd23e0497" />

<img width="1169" height="415" alt="image" src="https://github.com/user-attachments/assets/cd159da0-7c7a-4c70-ad4f-a8afc910fb62" />

可以觀察逐筆候選、ensemble後的成績，以及是否要剔除候選的過程

<img width="600" height="136" alt="image" src="https://github.com/user-attachments/assets/a4341d4a-6e0b-40fb-8e24-1fa9cb479a0a" />

---

## 11. 完整重現最終成績的建議方式

為避免助教或使用者因隨機性導致結果不一致，本專題保留了當時訓練完成的實驗資料夾：

`./output/BW_output/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057`

`./output/BW_output/D16_L5_H12_TrainOn2012-2014_TestOn2012-2014-_1114-134857`

若專案倉庫中已附上上述資料夾，則助教可以在 **不重新訓練** 的情況下完成評分與 ensemble，
以取得與報告中完全一致的數據。

### 11.1 不重新訓練，直接評分與 ensemble

確認上述兩個資料夾存在於 ./output/BW_output/ 之下。
使用 ensemble_csv.py：
指定上述兩個實驗所有 checkpoint 對應的 test_ret.csv，執行 ensemble。如是直接clone下git的檔案，則直接執行應能復現實驗結果，就是只會跑Greedy forward selection與Hill climbing algorithm湊ensemble組合，而不是重新訓練。

### 11.2 重新訓練（結果接近但不一定完全相同）

若助教希望從頭跑一次訓練流程，可依第 7 節的指令，
分別跑「實驗 A」與「實驗 B」，再依 8～10 節完成評分與ensemble。
由於訓練過程仍有隨機性，即使seed固定，實際數值可能略有差異，
但整體表現應會與原有結果接近。

## 12. 其他注意事項

1. 請確認環境有對應的 CUDA / 驅動程式。

2. seed： 程式中預設 --seed 42，但仍可能因環境差異造成些許差異。

3. 若出現缺少套件的錯誤訊息（例如 ModuleNotFoundError），請依照錯誤提示使用 pip install <package_name> 安裝對應套件。

