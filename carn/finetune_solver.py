import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
from sample import save_image
from collections import OrderedDict

import argparse
import sys
from pathlib import Path
import importlib.util
from swiftnet.evaluation import get_semseg_map, evaluate_semseg
from torchvision.transforms import Compose

from swiftnet.models.semseg import SemsegModel
from swiftnet.models.resnet.resnet_single_scale import *
from swiftnet.data.transform import *
from swiftnet.data.mux.transform import *
from swiftnet.data.cityscapes import Cityscapes
from swiftnet.models.util import get_n_params
from swiftnet.evaluation import StorePreds, StoreSubmissionPreds

from torch.autograd import Variable

import cv2
from PIL import Image as pimg

def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0:
            self.refiner = model(scale=cfg.scale,
                                 group=cfg.group)
            #state_dict = torch.load("./checkpoint/carn_10000.pth")
            #new_state_dict = OrderedDict()
            #for k, v in state_dict.items():
            #    name = k
                # name = k[7:] # remove "module."
            #    new_state_dict[name] = v

            #self.refiner.load_state_dict(new_state_dict)
        else:
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()
        elif cfg.loss_fn in ["CrossEntropyLoss"]:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = 19)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name, str(cfg.scale)))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
       
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr, hr_name, label = inputs[-1][0], inputs[-1][1], inputs[-1][2], inputs[-1][3]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                hr = hr.to(self.device)
                lr = lr.to(self.device)

                # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
                # print(lbl)
                # hr_name_str = ','.join(hr_name)
                # print(f'img list: {hr_name_str}')
                sr = refiner(lr, scale)
                # print(f'Size of HR: {hr.shape}')
                # print(f'Size of LR: {lr.shape}')
                print(f'Size of SR: {sr.shape}')
                # print(f'Type of SR: {sr.dtype}')
                # for i in range(len(hr_name)):
                #     sr_save = sr[i].detach().squeeze(0)
                #     sr_save_path = "/home/ubuntu/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/sr/{}".format(hr_name[i])
                #     save_image(sr_save, sr_save_path)
                # print("SR saved.")

                # pred_sr = []
                # pred_hr = []

                # for i in range(len(hr_name)):
                #     img_name = hr_name[i][:-16]
                #     # print(f'now img name: {img_name}')
                #     pred_maps = self.run_seg_inference(img_name)
                #     pred_sr.append(pred_maps[0].astype(np.float32))
                #     pred_hr.append(pred_maps[1].astype(np.float32))

                # pred_sr_array = np.stack(pred_sr)
                # pred_hr_array = np.stack(pred_hr)
                # pred_sr_tensor = torch.from_numpy(pred_sr_array)
                # pred_hr_tensor = torch.from_numpy(pred_hr_array)
                # # print(f'shape of pred_sr_tensor: {pred_sr_tensor.shape}')
                # pred_sr_tensor = Variable(pred_sr_tensor, requires_grad = True)
                # pred_hr_tensor = Variable(pred_hr_tensor, requires_grad = True)

                # if cfg.verbose and self.step % 100 == 0:
                #     sr_save = sr[0].detach().squeeze(0)
                #     print("sr_save size: {}".format(sr_save.size()))
                #     sr_im_path = "/home/ubuntu/cloudseg-project/CARN-pytorch/save/sr_{}.png".format(self.step)
                #     save_image(sr_save, sr_im_path)
                #     save_image(hr[0], "/home/ubuntu/cloudseg-project/CARN-pytorch/save/hr_{}.png".format(self.step))
                #     print("saved! step: {}".format(self.step))
                

                # ----- import the backend model here -----
                use_bn = True 
                resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
                model = SemsegModel(resnet, Cityscapes.num_classes, use_bn=use_bn)
                model.load_state_dict(torch.load('swiftnet/weights/swiftnet_ss_cs.pt'), strict=True)
                for param in model.parameters():
                    param.requires_grad = False
                model = model.cuda()

                logits_sr, additional_sr = model.forward([sr])
                logits_hr, additional_hr = model.forward([hr])


                # print(f'logits shape: {logits.shape}') # torch.Size([1, 19, 1024, 2048])
                # pred_sr_ori = nn.functional.gumbel_softmax(logits_sr, dim=1, hard=True)#.cpu().float()#.numpy().astype(np.float32)
                # pred_hr_ori = nn.functional.gumbel_softmax(logits_hr, dim=1, hard=True)#.cpu().float()#.numpy().astype(np.float32)
                
                # logits_sr = logits_sr.to(self.device) 
                # logits_hr = logits_hr.to(self.device)

                # soft_argmax = SoftArgmax1D()
                # for param in soft_argmax.parameters():
                #     param.requires_grad = False

                # print(f'shape of logits_sr: {logits_sr.size()}')
                # logits_sr.permute(0, 2, 3, 1)
                # logits_hr.permute(0, 2, 3, 1)

                # print(f'shape of logits_sr after permute: {logits_sr.size()}')

                # pred_sr = soft_argmax(logits_sr)
                # pred_hr = soft_argmax(logits_hr)

                # pred_sr_ori = torch.argmax(logits_sr, dim=1).float()
                # pred_hr_ori = torch.argmax(logits_hr, dim=1).float()

                # print(f'shape of pred_sr_ori: {pred_sr_ori.shape}')
                # print(pred_hr_ori)
                # print(pred_hr)

                # print(f'largest in softargmax: {pred_hr.max()}')

                
                # print(f'shape of pred_sr: {pred_sr.shape}')
                # print(pred_sr[0])

                # diff = pred_sr_ori - pred_sr
                
                # print(f'diff: {1024*2048 - (diff == 0).sum()}')

                # ----- output color -----
                # scale = 255
                # mean = Cityscapes.mean
                # std = Cityscapes.std
                # store_dir = 'swiftnet/configs/out/sr'
                # store_dir_color = 'swiftnet/configs/outc/sr'
                # store_dir_ori = 'swiftnet/configs/out/label'
                # store_dir_color_ori = 'swiftnet/configs/outc/label'
                # to_color = ColorizeLabels(Cityscapes.color_info)
                # # to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
                # sp1 = StoreSubmissionPreds(store_dir, lambda x: x, to_color, store_dir_color)
                # sp2 = StoreSubmissionPreds(store_dir_ori, lambda x: x, to_color, store_dir_color_ori)
                # sp1(pred_sr_ori.detach().cpu().numpy().astype(np.uint32))
                # sp2(label.byte().cpu().numpy().astype(np.uint32))
                # print('printed')


                # output:
                # shape of pred_sr: torch.Size([1, 19, 1024, 2048])
                
                # nan bug, fixed
                # nan_tensor = torch.isnan(pred_sr[0])
                # sum_sr = torch.sum(nan_tensor)
                # print(f'sum sr: {sum_sr}')
                # print(1024*2048)

                # print(pred_sr[0])
                # print(label)

                # unique, counts = np.unique(label, return_counts=True)
                label = label.to(self.device).long()
                # print(label)
                # print(f'label size: {label.shape}')
                # print(f'logits_sr size: {logits_sr.shape}')
                
                # print('label count: ')
                # print(dict(zip(unique, counts)))

                # loss = nn.functional.cross_entropy(logits_sr, label, ignore_index = 19), nn.functional.cross_entropy(logits_hr, label, ignore_index = 19)
                # self.writer.add_scalar("Cityscapes", loss.data[0], self.step)

                # new loss function
                # loss = self.loss_fn(logits_sr, label) - self.loss_fn(logits_hr, label)
                # loss = nn.functional.l1_loss(self.loss_fn(logits_sr, label), self.loss_fn(logits_hr, label)) * 0.08 + nn.functional.l1_loss(sr, hr) * 0.92
                loss = nn.functional.l1_loss(sr, hr) * 0.95 + self.loss_fn(logits_sr, label) * 0.05
                print(f'analytics-aware loss: {loss}')
                # self.writer.add_scalar("loss", loss, self.step)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                self.step += 1
                print("step {}".format(self.step))
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    print("evaluate step {}".format(self.step))
                    self.writer.add_scalar("loss", loss, self.step)
                    if cfg.scale > 0:
                        pass
                        # psnr = self.evaluate("dataset/Cityscapes", scale=cfg.scale, num_step=self.step)
                        # self.writer.add_scalar("Cityscapes", psnr, self.step)

                    else:    
                        psnr = [self.evaluate("dataset/Urban100", scale=i, num_step=self.step) for i in range(2, 5)]
                        self.writer.add_scalar("Urban100_2x", psnr[0], self.step)
                        self.writer.add_scalar("Urban100_3x", psnr[1], self.step)
                        self.writer.add_scalar("Urban100_4x", psnr[2], self.step)
                            
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    # def _softargmax(self, x, beta=1e10):
    #     x = tf.convert_to_tensor(x)
    #     x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    #     return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
    
    def evaluate(self, test_data_dir, scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        refiner.eval()

        test_data   = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            sr = refiner(lr_patch, scale).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            # mean_psnr += psnr(im1, im2) / len(test_data)

        return mean_psnr

    # def save_sr_evaluate(self, test_data_dir, scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        refiner.eval()

        test_data   = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            sr = refiner(lr_patch, scale).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]

    # def evaluate_seg(self, model, conf):
    #     conf_path = "swiftnet/configs/single_scale.py"
    #     conf = import_module(conf_path)
    #     for loader_val, loader_sr, name in conf.eval_loaders:
    #         iou, per_class_iou = evaluate_semseg(model, loader_val, class_info, observers=conf.eval_observers)
    #         print(f'{name}: {iou:.2f}')

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr

    def import_module(self, path):
        spec = importlib.util.spec_from_file_location("module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # def run_seg_inference(self, img_name):
    #     batch_size = 1 
    #     nw = 8 
    #     root = Path('swiftnet/datasets/Cityscapes')

    #     scale = 255
    #     mean = Cityscapes.mean
    #     std = Cityscapes.std
    #     mean_rgb = tuple(np.uint8(scale * np.array(mean)))

    #     num_levels = 1
    #     alphas = [1.]
    #     target_size = ts = (2048, 1024)
    #     target_size_feats = (ts[0] // 4, ts[1] // 4)

    #     eval_each = 4

    #     trans_train = trans_val = Compose(
    #         [Open(),
    #         RemapLabels(Cityscapes.map_to_id, Cityscapes.num_classes),
    #         Pyramid(alphas=alphas),
    #         SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
    #         Normalize(scale, mean, std),
    #         Tensor(),
    #         ]
    #     )

    #     dataset_single_val = Cityscapes(root, transforms=trans_val, subset='val', image_name=img_name)
    #     dataset_single_sr = Cityscapes(root, transforms=trans_val, subset='sr', image_name=img_name)
    #     loader_single_val = DataLoader(dataset_single_val, batch_size=batch_size, collate_fn=custom_collate, num_workers=nw)
    #     loader_single_sr = DataLoader(dataset_single_sr, batch_size=batch_size, collate_fn=custom_collate, num_workers=nw)

    #     use_bn = True 
    #     resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
    #     model = SemsegModel(resnet, Cityscapes.num_classes, use_bn=use_bn)
    #     model.load_state_dict(torch.load('swiftnet/weights/swiftnet_ss_cs.pt'), strict=True)

    #     total_params = get_n_params(model.parameters())
    #     ft_params = get_n_params(model.fine_tune_params())
    #     ran_params = get_n_params(model.random_init_params())
    #     spp_params = get_n_params(model.backbone.spp.parameters())
    #     assert total_params == (ft_params + ran_params)
    #     print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
    #     print(f'SPP params: {spp_params:,}')

    #     eval_loaders = [(loader_single_val, loader_single_sr, 'val')]
    #     eval_observers = []

    #     # conf_path = 'swiftnet/configs/single_image_single_scale.py'
    #     # conf = self.import_module(conf_path)

    #     class_info = dataset_single_val.class_info
    #     model = model.cuda()

    #     for loader_val, loader_sr, name in eval_loaders:
    #         # iou, per_class_iou = evaluate_semseg(model, loader_val, class_info, observers=eval_observers)
    #         # print(f'{name}: {iou:.2f}')
    #         pred_maps = get_semseg_map(model, loader_sr, loader_val, class_info)
    #         print(pred_maps[0].shape)
    #         print(pred_maps[1].shape)
    #         return pred_maps #sr then val


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
