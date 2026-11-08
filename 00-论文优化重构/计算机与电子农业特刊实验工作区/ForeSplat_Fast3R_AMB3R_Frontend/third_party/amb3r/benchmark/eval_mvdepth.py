import os
import sys
import torch
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

robustmvd_path = os.path.join(current_dir, "tools", "robustmvd")
if robustmvd_path not in sys.path:
    sys.path.insert(0, robustmvd_path)

from amb3r.model import AMB3R
from amb3r.model_zoo import load_model
from rmvd import create_evaluation, prepare_custom_model


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default="./outputs/mvdepth")
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--model_name', type=str, default="amb3r", choices=['amb3r', 'da3', 'omega', 'dvlt'])
    parser.add_argument('--ckpt_path', type=str, default="../checkpoints/amb3r.pt")
    return parser


@torch.no_grad()
def main(args):
    model = load_model(args.model_name, ckpt_path=args.ckpt_path)
    model = prepare_custom_model(model, num_gpus=1)

    
    metric_scale = args.metric
    results_path = args.results_path + ("_metric_scale" if metric_scale else "")


    evaluation = create_evaluation(
            evaluation_type="robustmvd",
            out_dir=results_path,
            eval_uncertainty=False,
            alignment=None if metric_scale else 'median',
        )

    if args.model_name == 'omega':
        eval_size = (384, 512)
    elif args.model_name == 'dvlt':
        eval_size = (378, 504)
    else:
        eval_size = (392, 518)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        results = evaluation(
                model=model,
                eth3d_size=eval_size,
                kitti_size=eval_size,
                dtu_size=eval_size,
                scannet_size=eval_size,
                tanks_and_temples_size=eval_size)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)


