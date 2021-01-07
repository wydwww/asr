# ASR

This repository contains part of the source code to implement Analytics-aware Super-Resolution (ASR) in *Enabling Edge-Cloud Video Analytics for Robotic Applications* ([Runespoor, INFOCOM '21](https://www.cse.ust.hk/~ywanggf/public/files/runespoor-infocom21.pdf)) and a previous workshop paper ([CloudSeg, HotCloud '19](https://www.cse.ust.hk/~ywanggf/public/files/cloudseg-hotcloud19.pdf)).
This repository is for archiving and provided as-is.
The final model weights are provided for evaluation, and you can use the code as a reference to develop related projects.
You might find many development-environment-related hardcoded settings. It might be inconvient that this legacy code does not have a proper configuration input method.

We implement ASR based on a super-resolution model [CARN](https://github.com/nmhkahn/CARN-pytorch) in PyTorch.
The semantic segmentation model used here is [SwiftNet](https://github.com/orsic/swiftnet).
The basic idea is to call the CV model (SwiftNet here) to train the super-resolution model to make it anaytics-aware and have better reconstruction performance for the CV task.
Please check those two repositories to better understand the code.

### Dataset
Please follow the instructions of CARN and SwiftNet to prepare the dataset ([Cityscapes](https://www.cityscapes-dataset.com/)).
The `dataset` directory should have a similar structure like:
```
dataset
└── Cityscapes
    ├── Cityscapes_train_HR
    ├── Cityscapes_train_LR_bicubic
    ├── Cityscapes_valid_HR
    ├── Cityscapes_valid_LR_bicubic (these data directories train the base SR)
    └── gtFine (this is the label of the semantic segmentation task)
```
`swiftnet/datasets/Cityscapes` requires symbolic links to the Cityscapes data, including `gtFine` and `img`.

### Pretrained Models
The pretrained base SR model and the ASR model are in the `best` directory.
Please follow the instruction of SwiftNet to download its pretrained weights.

### Code Structure
`carn/train.py` is the entry to the training code.
`carn/solver_vanilla_training.py` and `carn/finetune_solver.py` are loss functions used in different training stages (base and analytics-aware). Need to hand config in `carn/train.py`.

### Commands
An example command to train ASR:
```shell
$ python carn/train.py --patch_size 64 \
                       --batch_size 64 \
                       --max_steps 600000 \
                       --decay 400000 \
                       --model carn \
                       --ckpt_name carn \
                       --ckpt_dir checkpoint/carn \
                       --scale 4 \
                       --num_gpu 4 \
                       --print_interval 100 \
                       --loss_fn CrossEntropyLoss (add this argument only in analytics-aware training with finetune_solver.py)
```

Evaluation:
```shell
python carn/sample.py --model carn \
                      --test_data_dir dataset/Cityscapes \
                      --scale 4 \
                      --ckpt_path ./best/carn_4_vanilla.pth \
                      --sample_dir sample
```

### Citation
```
@inproceedings{wang2021enabling,
  title={Enabling Edge-Cloud Video Analytics for Robotic Applications},
  author={Wang, Yiding and Wang, Weiyan and Liu, Duowen and Jin, Xin and Jiang, Junchen and Chen, Kai},
  booktitle={IEEE INFOCOM 2021-IEEE Conference on Computer Communications},
  year={2021},
  organization={IEEE}
}
```