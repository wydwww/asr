from torch.utils.data import Dataset
from pathlib import Path

from .labels import labels

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
i = 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1

id_to_map = {id: i for i, id in map_to_id.items()}


class Cityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', image_name=None):
        self.root = root
        self.images_dir = self.root / 'img/left/leftImg8bit' / subset
        print(f'Image path: {self.images_dir}')
        # self.labels_dir = self.root / 'gtFine' / subset
        self.labels_dir = self.root / 'gtFine' / 'finetune_only'
        # if subset == 'sr':
            #fine tune only, use h5 dataset 800 images
            # self.labels_dir = self.root / 'gtFine' / 'finetune_only'
        self.subset = subset
        self.has_labels = subset != 'test'
        if image_name:
            if subset == 'sr':
                self.images = list(self.images_dir.glob('{}_leftImg8bit.png'.format(image_name)))
            else:
                self.images = list(self.images_dir.glob('{}_leftImg8bit.png'.format(image_name)))
        else:
            self.images = list(sorted(self.images_dir.glob('*/*.png')))
        if self.has_labels:
            if image_name:
                #remove */
                self.labels = list(self.labels_dir.glob('{}_gtFine_labelIds.png'.format(image_name)))
            else:
                self.labels = list(sorted(self.labels_dir.glob('*/*_gtFine_labelIds.png')))
        self.transforms = transforms
        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        if self.has_labels:
            ret_dict['labels'] = self.labels[item]
        return self.transforms(ret_dict)
