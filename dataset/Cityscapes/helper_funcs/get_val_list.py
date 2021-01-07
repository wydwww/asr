import os
import glob

img_path = 'Cityscapes_valid_LR_bicubic/X1800/'
mask_path = 'gtFine/'

img_list = glob.glob(img_path+'/*.png')

mask_list = [i.replace('Cityscapes_valid_LR_bicubic/X1800', 'gtFine').replace('leftImg8bit', 'gtFine_labelTrainIds') for i in img_list]

with open("cityscapes_val_list_900p_upsample.txt", "w") as text_file:
    for i in range(len(img_list)):
        print("{} {}".format(img_list[i], mask_list[i]), file=text_file)

