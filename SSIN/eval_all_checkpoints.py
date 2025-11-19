import os

from Trainer import MaskedTrainer
from utils.utils import Paths, init_seeds
import utils.config as cfg
from main_train import get_default_args, get_data_path


EXPERIMENT_DIR = ".\output\BW_output\main\D16_L4_H8_TrainOn2012-2014_TestOn2012-2014-_1114-041057"

RUN_PREFIX = "1114-041057"

DATASET = "bw"
D_K = 16     
D_MODEL = 32
N_LAYERS = 4     
N_HEAD = 8    


def build_trainer():
    parser = get_default_args()

    args = parser.parse_args([])

    args.dataset = DATASET
    args.sub_out_dir = "main"

    get_data_path(args)

    args.output_dir = EXPERIMENT_DIR

    args.d_k = D_K
    args.d_v = args.d_k
    args.d_model = D_MODEL 
    args.n_layers = N_LAYERS
    args.n_head = N_HEAD

    init_seeds(args)

    paths = Paths(args.output_dir)

    trainer = MaskedTrainer(args=args, out_path=paths, init_training=False)

    return trainer


def main():
    trainer = build_trainer()

    ckpt_dir = os.path.join(EXPERIMENT_DIR, "train", "checkpoints_path")
    ckpt_files = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".pyt") and f.startswith("checkpoint_")
    ]
    ckpt_files.sort()

    print(f"找到 {len(ckpt_files)} 個 checkpoint：")
    for f in ckpt_files:
        print("  ", f)

    for idx, ckpt_name in enumerate(ckpt_files, start=1):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        run_name = f"{RUN_PREFIX}_{idx:02d}"
        out_root = os.path.join(EXPERIMENT_DIR, run_name)
        test_dir = os.path.join(out_root, "test")
        os.makedirs(test_dir, exist_ok=True)

        csv_path = os.path.join(test_dir, "test_ret.csv")

        print(f"\n=== [{idx}/{len(ckpt_files)}] 測試 {ckpt_name}")
        print(f"    輸出到：{csv_path}")

        trainer.test(csv_path, model_path=ckpt_path)


if __name__ == "__main__":
    main()
