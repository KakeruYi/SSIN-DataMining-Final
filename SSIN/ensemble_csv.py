import os
import os.path as osp
from glob import glob
import random

import numpy as np
import pandas as pd

from postprocess.eval_methods import calc_scores


SMALL_DIR = ".\output\BW_output\main\D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057"
SMALL_PREFIX = "1114-041057"

BIG_DIR = ".\output\BW_output\main\D16_L5_H12_TrainOn2012-2014_TestOn2012-2014-_1114-134857"
BIG_PREFIX = "1114-134857"

# RMSE 小於等於這個門檻來做候選
RMSE_THRESH = 0.985

LABEL_THR = 0.0

RANDOM_SEED = 42

N_SINGLE_STARTS = 15    
N_MULTI_STARTS = 15 

RAND_K_MIN = 2
RAND_K_MAX = 8

# greedy / hill-climb
IMPROVE_TOL = 5e-5        # 需要比目前RMSE至少好多少才算「有進步」
MAX_SET_SIZE = 12         # ensemble最大成員數（避免太大）

OUT_ENSEMBLE_CSV = ".\output\BW_output\ensemble\\advanced_best_ensemble.csv"


def compute_scores_from_arrays(preds, labels, thr=0.0):
    preds = preds.copy()
    labels = labels.copy()

    if thr > 0:
        mask = labels >= thr
        preds = preds[mask]
        labels = labels[mask]

    preds[preds < 0] = 0.0
    rmse, mae, nse = calc_scores(preds, labels)
    return rmse, mae, nse


def load_all_checkpoints():
    models = {}

    def _load_group(base_dir, prefix, tag):
        pattern = osp.join(base_dir, f"{prefix}_*", "test", "test_ret.csv")
        paths = sorted(glob(pattern))
        if not paths:
            print(f"[警告] 找不到任何檔案：{pattern}")
            return []

        print(f"\n=== 載入 {tag} 組實驗，共 {len(paths)} 個 checkpoint ===")
        return paths

    small_paths = _load_group(SMALL_DIR, SMALL_PREFIX, "小模型 (L4/H8)")
    big_paths = _load_group(BIG_DIR, BIG_PREFIX, "大模型 (L5/H12)")

    all_paths = small_paths + big_paths
    if not all_paths:
        raise RuntimeError("完全沒有找到任何 test_ret.csv，請檢查路徑設定。")

    base_df = None
    base_labels = None

    for path in all_paths:
        df = pd.read_csv(path)

        # 判斷 label 欄位名稱
        if "rainfall" in df.columns.to_list():
            label_col = "rainfall"
        elif "label" in df.columns.to_list():
            label_col = "label"
        else:
            raise TypeError(f"{path} 找不到 rainfall 或 label 欄位")

        labels = df[label_col].astype(float).values
        preds = df["pred"].fillna(0).astype(float).values

        if base_df is None:
            base_df = df.copy()
            base_labels = labels
        else:
            if len(df) != len(base_df):
                raise ValueError(f"{path} 長度 {len(df)} 與 base_df {len(base_df)} 不一致")

        rmse, mae, nse = compute_scores_from_arrays(preds, labels, thr=LABEL_THR)

        ckpt_dir = osp.dirname(osp.dirname(path))
        folder_name = osp.basename(ckpt_dir) 

        if folder_name.startswith(SMALL_PREFIX):
            idx = folder_name.split("_")[-1]   
            model_name = "small_" + idx        
        elif folder_name.startswith(BIG_PREFIX):
            idx = folder_name.split("_")[-1]
            model_name = "big_" + idx          
        else:
            model_name = folder_name   

        models[model_name] = {
            "path": path,
            "preds": preds,
            "labels": labels,
            "rmse": rmse,
            "mae": mae,
            "nse": nse,
        }

        print(f"{model_name:10s} | RMSE={rmse:.4f}, MAE={mae:.4f}, NSE={nse:.4f} | {path}")

    return models, base_df, base_labels


def evaluate_combo(combo, models, base_labels):
    preds_list = [models[name]["preds"] for name in combo]
    preds_stack = np.stack(preds_list, axis=0)
    ens_preds = preds_stack.mean(axis=0)
    rmse, mae, nse = compute_scores_from_arrays(ens_preds, base_labels, thr=LABEL_THR)
    return rmse, mae, nse, ens_preds


def greedy_forward_selection(candidates, models, base_labels, start_combo=None):
    """
    Greedy forward selection. 不斷嘗試加入一個新的candidate看RMSE是否有明顯下降
    """
    if start_combo is None:
        current = []
    else:
        current = list(start_combo)

    if len(current) == 0:
        current_rmse = float("inf")
        current_mae = None
        current_nse = None
    else:
        current_rmse, current_mae, current_nse, _ = evaluate_combo(current, models, base_labels)

    improved = True
    while improved and len(current) < MAX_SET_SIZE:
        improved = False
        best_local_rmse = current_rmse
        best_local_mae = current_mae
        best_local_nse = current_nse
        best_to_add = None

        remaining = [c for c in candidates if c not in current]
        for cand in remaining:
            tmp_combo = current + [cand]
            rmse, mae, nse, _ = evaluate_combo(tmp_combo, models, base_labels)
            if rmse + IMPROVE_TOL < best_local_rmse:
                best_local_rmse = rmse
                best_local_mae = mae
                best_local_nse = nse
                best_to_add = cand

        if best_to_add is not None:
            current.append(best_to_add)
            current_rmse = best_local_rmse
            current_mae = best_local_mae
            current_nse = best_local_nse
            improved = True
            print(f"[Greedy] 加入 {best_to_add}, set size={len(current)}, RMSE={current_rmse:.4f}, NSE={current_nse:.4f}")

    return current, current_rmse, current_mae, current_nse


def hill_climb_local(candidates, models, base_labels, start_combo):
    """
    hill-climbing. 在目前的set做「加一個 / 減一個 / 換一個」鄰居搜尋. 只要找到更好的RMSE就移動過去
    """
    current = list(start_combo)
    current_rmse, current_mae, current_nse, _ = evaluate_combo(current, models, base_labels)

    while True:
        best_neighbor_rmse = current_rmse
        best_neighbor_mae = current_mae
        best_neighbor_nse = current_nse
        best_neighbor = None

        current_set = set(current)
        others = [c for c in candidates if c not in current_set]

        if len(current) < MAX_SET_SIZE:
            for cand in others:
                tmp = current + [cand]
                rmse, mae, nse, _ = evaluate_combo(tmp, models, base_labels)
                if rmse + IMPROVE_TOL < best_neighbor_rmse:
                    best_neighbor_rmse = rmse
                    best_neighbor_mae = mae
                    best_neighbor_nse = nse
                    best_neighbor = tmp

        if len(current) > 1:
            for i in range(len(current)):
                tmp = current[:i] + current[i+1:]
                rmse, mae, nse, _ = evaluate_combo(tmp, models, base_labels)
                if rmse + IMPROVE_TOL < best_neighbor_rmse:
                    best_neighbor_rmse = rmse
                    best_neighbor_mae = mae
                    best_neighbor_nse = nse
                    best_neighbor = tmp

        if len(current) >= 1 and len(others) >= 1:
            for i in range(len(current)):
                for cand in others:
                    tmp = current[:i] + current[i+1:] + [cand]
                    rmse, mae, nse, _ = evaluate_combo(tmp, models, base_labels)
                    if rmse + IMPROVE_TOL < best_neighbor_rmse:
                        best_neighbor_rmse = rmse
                        best_neighbor_mae = mae
                        best_neighbor_nse = nse
                        best_neighbor = tmp

        if best_neighbor is None:
            break
        else:
            current = best_neighbor
            current_rmse = best_neighbor_rmse
            current_mae = best_neighbor_mae
            current_nse = best_neighbor_nse
            print(f"[HillClimb] 更新組合 size={len(current)}, RMSE={current_rmse:.4f}, NSE={current_nse:.4f}")

    return current, current_rmse, current_mae, current_nse


def save_best_ensemble_csv(best_combo, models, base_df, base_labels, out_path):
    """
    根據最佳組合，輸出 ensemble 後的 test_ret.csv。
    """
    rmse, mae, nse, ens_preds = evaluate_combo(best_combo, models, base_labels)

    df = base_df.copy()
    df["pred"] = ens_preds

    if "rainfall" in df.columns.to_list():
        label_col = "rainfall"
    elif "label" in df.columns.to_list():
        label_col = "label"
    else:
        raise TypeError("base_df 找不到 rainfall 或 label 欄位")

    rmse2, mae2, nse2 = compute_scores_from_arrays(df["pred"].values, df[label_col].values, thr=LABEL_THR)
    print("\n[Best Ensemble 最終確認] - RMSE={:.4f}, MAE={:.4f}, NSE={:.4f}".format(rmse2, mae2, nse2))

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"已將最佳 ensemble 結果輸出到：{out_path}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    models, base_df, base_labels = load_all_checkpoints()

    candidates = [name for name, m in models.items() if m["rmse"] <= RMSE_THRESH]

    print("\n=== 候選 checkpoint（RMSE <= {:.4f}）共有 {} 個 ===".format(RMSE_THRESH, len(candidates)))
    for name in sorted(candidates):
        m = models[name]
        print(f"{name:10s} | RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, NSE={m['nse']:.4f}")

    if len(candidates) == 0:
        raise RuntimeError("沒有任何 checkpoint 的 RMSE 小於等於 RMSE_THRESH，請調整門檻。")

    best_single_name = min(candidates, key=lambda n: models[n]["rmse"])
    print(f"\n最佳單一模型：{best_single_name}, RMSE={models[best_single_name]['rmse']:.4f}, NSE={models[best_single_name]['nse']:.4f}")

    start_combos = []

    start_combos.append([best_single_name])

    others = [n for n in candidates if n != best_single_name]
    n_single = min(N_SINGLE_STARTS, len(others))
    if n_single > 0:
        random_single = random.sample(others, n_single)
        for n in random_single:
            start_combos.append([n])

    seen = set()
    seen.update(tuple(sorted(c)) for c in start_combos)

    for _ in range(N_MULTI_STARTS):
        k = random.randint(RAND_K_MIN, min(RAND_K_MAX, len(candidates)))
        combo = sorted(random.sample(candidates, k))
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            start_combos.append(combo)

    known_good = ["big_25", "big_15", "big_10", "small_06", "small_07", "small_05"]
    if all(name in candidates for name in known_good):
        key = tuple(sorted(known_good))
        if key not in seen:
            start_combos.append(known_good)
            seen.add(key)
        print("\n已將已知不錯的 6-model 組合加入起始點：", known_good)
    else:
        print("\n已知不錯的 6-model 組合有些不在 candidates 裡，略過。")

    print("\n=== 總共有 {} 個起始組合要做 greedy + hill-climb ===".format(len(start_combos)))

    global_best_combo = None
    global_best_rmse = float("inf")
    global_best_mae = None
    global_best_nse = None

    for idx, combo in enumerate(start_combos, 1):
        print("\n==============================")
        print(f"起始組合 #{idx}: {combo}")
        print("==============================")

        g_combo, g_rmse, g_mae, g_nse = greedy_forward_selection(candidates, models, base_labels, start_combo=combo)
        print(f"[Greedy 結果] set={g_combo}, RMSE={g_rmse:.4f}, MAE={g_mae:.4f}, NSE={g_nse:.4f}")

        hc_combo, hc_rmse, hc_mae, hc_nse = hill_climb_local(candidates, models, base_labels, start_combo=g_combo)
        print(f"[HillClimb 結果] set={hc_combo}, RMSE={hc_rmse:.4f}, MAE={hc_mae:.4f}, NSE={hc_nse:.4f}")

        if hc_rmse < global_best_rmse or (np.isclose(hc_rmse, global_best_rmse) and (global_best_nse is None or hc_nse > global_best_nse)):
            global_best_rmse = hc_rmse
            global_best_mae = hc_mae
            global_best_nse = hc_nse
            global_best_combo = hc_combo
            print("\n>>> [更新 Global Best] 組合 =", global_best_combo)
            print("    RMSE={:.4f}, MAE={:.4f}, NSE={:.4f}".format(global_best_rmse, global_best_mae, global_best_nse))

    print("\n==============================")
    print("搜尋結束！最終 Global Best：")
    print("組合 =", global_best_combo)
    print("RMSE={:.4f}, MAE={:.4f}, NSE={:.4f}".format(global_best_rmse, global_best_mae, global_best_nse))
    print("==============================")

    save_best_ensemble_csv(global_best_combo, models, base_df, base_labels, OUT_ENSEMBLE_CSV)


if __name__ == "__main__":
    main()
