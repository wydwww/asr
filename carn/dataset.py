import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path

# from swiftnet.data.cityscapes import Cityscapes
from swiftnet.data.cityscapes.labels import labels

map_to_id = {}
i = 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1

id_to_map = {id: i for i, id in map_to_id.items()}

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()

class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.labels_dir = "dataset/Cityscapes/gtFine_train/"

        self.size = size
        h5f = h5py.File(path, "r")

        # get the name of image
        self.hr_name = [v[:] for v in h5f["HR"].keys()]
        
        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()

        # self.label = [self.labels_dir.glob(v.replace("leftImg8bit", "gtFine_labelIds")) for v in self.hr_name]
        self.label = [self.labels_dir + v.replace("leftImg8bit", "gtFine_labelIds") for v in self.hr_name]
        # print(self.label)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # self.class_info = [label.name for label in labels if label.ignoreInEval is False]
        # self.color_info = [label.color for label in labels if label.ignoreInEval is False]
    
        # self.color_info += [[0, 0, 0]]
    
        self.map_to_id = map_to_id
    
        self.id_to_map = {id: i for i, id in self.map_to_id.items()}
    
        self.num_classes = 19
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]

    def _process_label(self, label_path):
        # open
        label_open = np.array(Image.open(label_path))
        #remaplabel
        total = 35
        mapping = np.ones((total + 1,), dtype=np.uint8) * self.num_classes
        ignore_id = self.num_classes
        for i in range(len(mapping)):
            mapping[i] = self.map_to_id[i] if i in self.map_to_id else self.num_classes
        labels_trans = mapping[label_open].astype(label_open.dtype)
        # print(labels_trans)
        # unique, counts = np.unique(labels_trans, return_counts=True)
        # print('label count: ')
        # print(dict(zip(unique, counts)))
        # print(f'label size: {sum(counts)}')
        return labels_trans


    def __getitem__(self, index):
        size = self.size

        item = [(self.hr[index], self.lr[i][index], self.hr_name[index], self.label[index]) for i, _ in enumerate(self.lr)]
        # for finetune with whole image
        # item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        # item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        return [(self.transform(hr), self.transform(lr), hr_name, self._process_label(label)) for hr, lr, hr_name, label in item]

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        
        # print(self.name)
        #if "DIV" in self.name:
        if self.name == "Cityscapes":
            self.hr = glob.glob(os.path.join(dirname, "{}_valid_HR".format("Cityscapes"), "*.png"))
            self.lr = glob.glob(os.path.join(dirname, "{}_valid_LR_bicubic".format("Cityscapes"), 
                                             "X{}/*.png".format(scale)))
        elif self.name == "visdrone":
            self.hr = glob.glob(os.path.join(dirname, "{}_valid_HR".format("visdrone"), "*.jpg"))
            self.lr = glob.glob(os.path.join(dirname, "{}_valid_LR_bicubic".format("visdrone"), 
                                             "X{}/*.jpg".format(scale)))
        else:
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)

class TrainDatasetVisdrone(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDatasetVisdrone, self).__init__()

        self.labels_dir = "dataset/visdrone/visdrone_train_HR_label/"

        self.size = size
        h5f = h5py.File(path, "r")

        # get the name of image
        self.hr_name = [v[:] for v in h5f["HR"].keys()]
        
        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()

        # self.label = [self.labels_dir.glob(v.replace("leftImg8bit", "gtFine_labelIds")) for v in self.hr_name]
        self.label = [self.labels_dir + v.replace("jpg", "txt") for v in self.hr_name]
        # print(self.label)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        # item = [(self.hr[index], self.lr[i][index], self.hr_name[index], self.label[index]) for i, _ in enumerate(self.lr)]
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        # for finetune with whole image
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        # return [(self.transform(hr), self.transform(lr), hr_name, label) for hr, lr, hr_name, label in item]
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]


    def __len__(self):
        return len(self.hr)
