import os
import glob
import h5py
import scipy.misc as misc
import numpy as np

dataset_dir = "Cityscapes/"
dataset_type = "train"

f = h5py.File("Cityscapes_{}_X4.h5".format(dataset_type), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "X4"]:
    if subdir in ["HR"]:
        im_paths = glob.glob(os.path.join(dataset_dir, 
                                          "Cityscapes_{}_HR".format(dataset_type), 
                                          "*.png"))

    else:
        im_paths = glob.glob(os.path.join(dataset_dir, 
                                          "Cityscapes_{}_LR_bicubic".format(dataset_type), 
                                          subdir, "*.png"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = misc.imread(path)
        print(path)
        # save path to file
        grp.create_dataset(os.path.basename(path), data=im)
