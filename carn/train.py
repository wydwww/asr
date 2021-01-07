import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
import torch
from collections import OrderedDict
# from finetune_solver import Solver
from solver_vanilla_training import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_name", type=str)
    
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--train_data_path", type=str, 
                        default="dataset/Cityscapes_train.h5")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint")
    parser.add_argument("--sample_dir", type=str,
                        default="sample/")
    
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--group", type=int, default=1)
    
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1", "CrossEntropyLoss"], default="L1")

    return parser.parse_args()

def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    # fine tune
    #state_dict = torch.load("./checkpoint/carn/carn_7000.pth")
    #new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k
        # name = k[7:] # remove "module."
    #    new_state_dict[name] = v

    #net.load_state_dict(new_state_dict)

    
    solver = Solver(net, cfg)
    # for finetuning
    # solver.load("./checkpoint/visdrone/4/carn_1600.pth")
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
