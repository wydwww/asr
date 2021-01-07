from PIL import Image
import glob
import numpy as np
import skimage.measure as measure
from pathlib import Path
import sys
from tqdm import tqdm

def im2double(im):
        min_val, max_val = 0, 255
        out = (np.asarray(im)-min_val) / (max_val-min_val)
        return out

def psnr(im1, im2):
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr

def ssim(im1, im2):
    im1 = im2double(im1)
    im2 = im2double(im2)
    ssim = measure.compare_ssim(im1, im2, multichannel=True)
    return ssim

#hr_path = '/home/ubuntu/sr/CARN-pytorch/dataset/Cityscapes/Cityscapes_valid_HR'
#lr_path = '/home/ubuntu/sr/CARN-pytorch/dataset/Cityscapes/Cityscapes_valid_LR_bicubic/X0.25'

hr_path = '/home/ubuntu/cloudseg-project/CARN-pytorch/dataset/Cityscapes/Cityscapes_valid_HR'
lr_path = '/home/ubuntu/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val_x6_upscale'

hr_list = glob.glob(hr_path+'/*.png')
#lr_list = [i.replace(hr_path, lr_path)+ for i in hr_list]
lr_list = []
#for filename in glob.glob(lr_path+'/**/*.png'):
#    lr_list.append(filename)

for i in hr_list:
    if i.split('/')[-1][:3] == 'mun':
        lr_list.append(i.replace(hr_path, lr_path+'/munster'))
    elif i.split('/')[-1][:3] == 'fra':
        lr_list.append(i.replace(hr_path, lr_path+'/frankfurt'))
    elif i.split('/')[-1][:3] == 'lin':
        lr_list.append(i.replace(hr_path, lr_path+'/lindau'))

psnr_all = 0
ssim_all = 0
for i in tqdm(range(len(hr_list))):
    psnr_all += psnr(Image.open(hr_list[i]), Image.open(lr_list[i]))
    ssim_all += ssim(Image.open(hr_list[i]), Image.open(lr_list[i]))
print('psnr: {}'.format(psnr_all/len(hr_list)))
print('ssim: {}'.format(ssim_all/len(hr_list)))
