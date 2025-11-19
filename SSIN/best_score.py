import os
import os.path as osp
from glob import glob

import numpy as np
import pandas as pd

from postprocess.eval_methods import calc_scores


def print_scores(df, label_col, pred_col):

    df.loc[df[pred_col] < 0, pred_col] = 0

    predicts = df[pred_col].values
    labels = df[label_col].values
    rmse, mae, nse = calc_scores(predicts, labels)
    print("[Tot Result] - RMSE: {:.4f}, MAE: {:.4f}, NSE: {:.4f}".format(rmse, mae, nse))
    return rmse, mae, nse


def eval_one_csv(csv_path, thr=0.0):
    print(csv_path)
    ret_df = pd.read_csv(csv_path)
    print("Length: ", len(ret_df))


    pred_col = "pred"

    ret_df[pred_col] = ret_df[pred_col].fillna(0)
    ret_df[pred_col] = ret_df[pred_col].astype(float)


    if "rainfall" in ret_df.columns.to_list():
        label_col = "rainfall"
    elif "label" in ret_df.columns.to_list():
        label_col = "label"
    else:
        raise TypeError("Wrong label name!")


    if thr > 0:
        ret_df = ret_df[ret_df[label_col] >= thr]


    pred_cols = [c for c in ret_df.columns.values.tolist() if "pred" in c]
    results = {}
    for col in pred_cols:
        print(f"  column: {col}")
        rmse, mae, nse = print_scores(ret_df, label_col, col)
        results[col] = (rmse, mae, nse)

    return results


def main():

    EXPERIMENT_DIR = ".\output\BW_output\main\D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057"
    RUN_PREFIX = "1114-041057"  # 子資料夾 prefix

    pattern = osp.join(EXPERIMENT_DIR, f"{RUN_PREFIX}_*", "test", "test_ret.csv")
    csv_paths = sorted(glob(pattern))

    if not csv_paths:
        print("找不到任何 test_ret.csv，請確認 EXPERIMENT_DIR / RUN_PREFIX 是否正確。")
        return

    print(f"找到 {len(csv_paths)} 個 test_ret.csv\n")

    best_info = None  # (idx, path, rmse, mae, nse)

    for idx, csv_path in enumerate(csv_paths, start=1):
        print(f"==== Checkpoint #{idx} ====")
        res_dict = eval_one_csv(csv_path, thr=0.0)

        (rmse, mae, nse) = list(res_dict.values())[0]

        if best_info is None:
            best_info = (idx, csv_path, rmse, mae, nse)
        else:
            _, _, best_rmse, _, best_nse = best_info
            if rmse < best_rmse or (np.isclose(rmse, best_rmse) and nse > best_nse):
                best_info = (idx, csv_path, rmse, mae, nse)

        print()


    print("==== 最佳 checkpoint ====")
    best_idx, best_path, best_rmse, best_mae, best_nse = best_info
    print(f"Checkpoint index: {best_idx}")
    print(f"Path: {best_path}")
    print("[Best Result] - RMSE: {:.4f}, MAE: {:.4f}, NSE: {:.4f}".format(best_rmse, best_mae, best_nse))


if __name__ == "__main__":
    main()
