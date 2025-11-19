# SSIN-DataMining-Final
Optimize the SSIN by adjusting its architecture and parameters to improve performance. For academic research use only.

## 1. 專案說明

本專題以原始 **SSIN**（Spatio-Temporal Sequence Imputation Network）為基礎，  
針對模型結構與超參數（特別是 Transformer Encoder 的層數與 multi-head 數量）進行調整，  
並在 **PEMS-BW**（程式中以 `bw` 表示）資料集上完成訓練、評估與 ensemble，  
最終在課程指定的評分指標上達到約 **0.9665** 的表現。

本專案主要修改與使用的檔案包含：

- `main_train.py`：主訓練腳本（本次專題有調整部分 SSIN source code）
- `show_scores.py`：對單一實驗結果（`test_ret.csv`）計算評分指標
- `eval_all_checkpoints.py`：自動對某一實驗底下所有 checkpoint 做 testing，產生多個 `test_ret.csv`
- `ensemble_csv.py`：對多個 checkpoint 的 `test_ret.csv` 做 ensemble，尋找較佳組合

---

## 2. 環境需求與安裝

### 2.1 環境需求（建議）

- 作業系統：Windows / Linux / macOS 皆可
- Python：3.x（建議 3.8 以上）
- 必要套件（常見組合）：
  - `torch`（需配合實際 GPU / CUDA 版本安裝）
  - `numpy`
  - `pandas`
  - 其他依程式 import 而定

> 若專案內有 `requirements.txt`，可直接：
> ```bash
> pip install -r requirements.txt
> ```

### 2.2 取得專案

```bash
git clone <本專案的 GitHub 連結>
cd SSIN  # 以下指令預設在專案根目錄下執行

##3. 資料準備

本專題的實驗以 BW 資料集 為主（對應 --dataset bw）。

請確認資料夾結構如下（相對於專案根目錄）：

./data/
└── BW_132_data/
    └── pkl_data/
        ├── train/
        │   └── 2012-2014_data.pkl
        └── test/
            └── 2012-2014_data.pkl


main_train.py 會根據 --dataset 自動設定路徑（以 bw 為例）：

訓練資料：./data/BW_132_data/pkl_data/train/2012-2014_data.pkl

測試資料：./data/BW_132_data/pkl_data/test/2012-2014_data.pkl

若資料不在上述路徑，請依情況調整資料夾或程式碼。

##4. 專案目錄與輸出結構

訓練完成後，輸出會放在：

./output/BW_output/<sub_out_dir>/<實驗資料夾>/


預設 sub_out_dir = "main"，因此路徑通常為：

./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_<時間戳>/


一個典型的實驗資料夾內部結構會類似：

./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057/
├── args.json                     # 參數設定記錄
├── train/
│   └── checkpoints_path/         # 每 10 個 epoch 儲存一個 checkpoint
│       ├── ckpt_epoch_10.pth
│       ├── ckpt_epoch_20.pth
│       └── ...
└── test/
    ├── test_ret.csv              # 全部 epoch 跑完後的最終 testing 結果
    └── 其他中間輸出（如有）


注意：

每次執行 main_train.py 會依照當下時間產生新的 <時間戳> 實驗資料夾，不會覆蓋舊結果。

若要做 ensemble 或重新計算分數，需要知道欲使用的實驗資料夾路徑。

##5. 實驗設定總覽（本專題最終結果）

本專題最終報告採用兩組關鍵實驗設定，皆以 dataset=bw、model_type=SpaFormer 為主，
主要差異在於 Transformer Encoder 的層數與 head 數，以及 embedding 維度：

實驗 A

n_layers = 4

n_head = 8

d_model = 32

實驗 B

n_layers = 5

n_head = 12

d_model = 64

以下會示範如何在終端機下正確給參數，重現這兩組設定。

##6. 使用流程總覽

完整操作流程建議依照下列順序：

準備環境與資料（見上方第 2、3 節）

執行單一模型訓練：使用 main_train.py 跑「實驗 A」與「實驗 B」

使用 show_scores.py 對單一實驗做評分

使用 eval_all_checkpoints.py 對某一實驗底下所有 checkpoint 做 testing

使用 ensemble_csv.py 對多個 checkpoint 的結果做 ensemble，取得最終約 0.9665 的成績

（選擇性）直接使用我已保留的實驗輸出做評分與 ensemble，以完全重現報告數值

以下逐步說明。

##7. 單一模型訓練：main_train.py
###7.1 main_train.py 重要參數說明（節錄）

在程式開頭中，可看到主要參數：

資料相關

--dataset：bw / hk / bay（本專題使用 bw）

--sub_out_dir：輸出次資料夾名稱，預設 main

模型相關

--model_type：預設 "SpaFormer"

--n_layers：Transformer encoder layer 數

--n_head：multi-head attention 的 head 數

--d_k / --d_v：key/value 維度（程式中 d_v = d_k）

--d_model：embedding 維度（本專題重點調整參數之一）

訓練相關

--epochs：訓練總 epoch 數，預設 100（若有調整，請依實際狀況）

--gpu_id：使用的 GPU ID（字串），例如 "0"

--seed：隨機種子，預設 42

###7.2 實驗 A：n_layers=4, n_head=8, d_model=32

請先切到專案根目錄：

cd SSIN


執行指令範例：

python main_train.py \
  --dataset bw \
  --model_type SpaFormer \
  --n_layers 4 \
  --n_head 8 \
  --d_model 32 \
  --gpu_id 0 \
  --suffix expA


說明：

--dataset bw：指定使用 BW 資料集

--n_layers 4、--n_head 8、--d_model 32：對應「實驗 A」

--gpu_id 0：使用第 0 張 GPU（若只有一張 GPU，一般也是 0）

--suffix expA：方便在輸出資料夾名稱後看到此實驗的標記（可自行更改或省略）

訓練結束後，會在類似下列路徑產生實驗資料夾：

./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-expA_1114-041057/


（實際時間戳會依執行當下時間而異）

###7.3 實驗 B：n_layers=5, n_head=12, d_model=64

同樣在專案根目錄下執行：

python main_train.py \
  --dataset bw \
  --model_type SpaFormer \
  --n_layers 5 \
  --n_head 12 \
  --d_model 64 \
  --gpu_id 0 \
  --suffix expB


訓練完成後，會產生類似路徑：

./output/BW_output/main/D16_L5_H12_TrainOn2012-2014_TestOn2012-2014-expB_1114-134857/


小提醒：

請確認執行時終端機有看到「Training start...」等訊息，確保訓練正常進行。

訓練結束後，對應實驗資料夾內 test/test_ret.csv 即為該模型的最終 testing 結果。

##8. 計算評分：show_scores.py

當某一實驗訓練完成後，會在該實驗資料夾下的 test 子資料夾產生：

<實驗資料夾>/test/test_ret.csv


接著即可使用 show_scores.py 計算該 test_ret.csv 的評分指標。

###8.1 執行方式

在專案根目錄下執行：

python show_scores.py


此腳本會依內部實作，讀取對應的 test_ret.csv，並在終端機中輸出各項指標（如 MAE、RMSE 等，實際項目依原始 SSIN 評分程式而定）。

若 show_scores.py 需要指定路徑或額外參數，請依檔案內註解或程式碼為最終準則。
本 README 的目的為提示使用流程與時機：訓練完 → 產生 test_ret.csv → 執行 show_scores.py。

##9. 檢驗所有 checkpoints：eval_all_checkpoints.py

由於我們不預先知道哪一個 epoch 的表現最好，
因此撰寫了 eval_all_checkpoints.py 來自動對 某一實驗資料夾下的所有 checkpoint 逐一做 testing。

###9.1 使用時機

先完成「實驗 A」的訓練（或「實驗 B」）。

在 <實驗資料夾>/train/checkpoints_path/ 中，通常每 10 個 epoch 會儲存一個 checkpoint，例如：

ckpt_epoch_10.pth

ckpt_epoch_20.pth

...

執行 eval_all_checkpoints.py，讓它對這些 checkpoint 逐一進行 testing，並在對應的路徑產生多個 test_ret.csv。

###9.2 注意事項

程式中通常會在 eval_all_checkpoints.py 裡面寫死一個實驗路徑，請先打開檔案，找到路徑設定，例如（示意）：

EXP_DIR = "./output/BW_output/main/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057"


請將此路徑改成你要檢驗的實驗資料夾。

修改完成後，在專案根目錄執行：

python eval_all_checkpoints.py


執行完後，該實驗底下會對應產生多個 test_ret.csv（依照腳本設計，可能按 epoch 或 checkpoint 命名），
之後可再搭配 show_scores.py 或 ensemble_csv.py 做分析。

重要：

做完「實驗 A」後，請先將 eval_all_checkpoints.py 內的路徑設定為「實驗 A」的資料夾並跑一次。

再修改為「實驗 B」的資料夾路徑，再跑一次。
如此兩組實驗的所有 checkpoint 都會產生對應的 test_ret.csv，方便後續 ensemble。

##10. Ensemble：ensemble_csv.py

在取得多個 checkpoint 對應的 test_ret.csv 後，
可以透過 ensemble_csv.py 對不同組合的 checkpoint 結果進行 ensemble，
尋找更佳的整體表現。最終本專題透過該流程得到約 0.9665 的成績。

###10.1 使用流程

確定「實驗 A」與「實驗 B」底下的所有 checkpoint 都已經由 eval_all_checkpoints.py 產生對應的 test_ret.csv。

打開 ensemble_csv.py，根據檔案內註解：

指定要讀取的 test_ret.csv 路徑（可能是多個實驗資料夾、多個 checkpoint）

檢查 ensemble 的策略與組合方式（例如簡單平均、依權重加權等，請以程式實作為準）

在專案根目錄執行：

python ensemble_csv.py


程式會逐一測試各種 checkpoint 組合進行 ensemble，輸出對應分數，
並可找到最終表現約 0.9665 的組合。

由於實際 ensemble 策略詳細實作寫在 ensemble_csv.py 內，
若需了解演算法細節（例如是否為簡單平均、如何選擇組合等），
請直接參考該檔案中的程式碼與註解。

###11. 完整重現最終成績的建議方式

為避免助教或使用者因隨機性導致結果不一致，本專題特別保留了當時訓練完成的實驗資料夾：

./output/BW_output/D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057

./output/BW_output/D16_L5_H12_TrainOn2012-2014_TestOn2012-2014-_1114-134857

若專案倉庫中已附上上述資料夾，則助教可以在 不重新訓練 的情況下完成評分與 ensemble，
以取得與報告中完全一致的數據。

###11.1 不重新訓練，直接評分與 ensemble

確認上述兩個資料夾存在於 ./output/BW_output/ 之下。

使用 show_scores.py：

依檔案內邏輯，讓腳本讀取 test_ret.csv 計算分數。

使用 eval_all_checkpoints.py：

分別將實驗路徑設定為上述兩個資料夾，一次處理一個實驗。

使用 ensemble_csv.py：

指定上述兩個實驗所有 checkpoint 對應的 test_ret.csv，執行 ensemble。

###11.2 重新訓練（結果接近但不一定完全相同）

若助教希望從頭跑一次訓練流程，可依第 7 節的指令，
分別跑「實驗 A」與「實驗 B」，再依 8～10 節完成評分與 ensemble。
由於訓練過程仍有隨機性，即使 seed 固定，實際數值可能略有差異，
但整體表現應會與原有結果接近。

##12. 其他注意事項

GPU 使用：

main_train.py 透過 --gpu_id 指定使用哪一張 GPU，請確認環境有對應的 CUDA / 驅動程式。

若在 CPU 上執行，訓練時間可能會非常長。

隨機種子：

程式中預設 --seed 42，有助於結果重現性，但仍可能因環境差異造成些許差異。

輸出不會覆蓋舊實驗：

由於輸出資料夾名稱中包含時間戳，每次訓練會建立新資料夾，
不會覆蓋先前的結果，有利於後續比較與 ensemble。

錯誤排除：

若出現缺少套件的錯誤訊息（例如 ModuleNotFoundError），
請依照錯誤提示使用 pip install <package_name> 安裝對應套件。

若依照本 README 的步驟操作，
助教應可順利完成：

在 BW 資料集上重現兩組關鍵實驗（n_layers=4, n_head=8, d_model=32 及 n_layers=5, n_head=12, d_model=64）

對單一實驗的結果進行評分

對所有 checkpoint 進行自動 testing

使用 ensemble 技巧得到最終約 0.9665 的成績

以利檢閱與成績評分。
