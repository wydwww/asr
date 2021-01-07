from PIL import Image
import os, sys
from tqdm import tqdm
import time

def down_sample(image_path, ratio):
    image = Image.open(image_path)
    width, height = image.size
    # width: 2048
    lr = image.resize((int(width/ratio), int(height/ratio)), Image.BILINEAR)
    #.resize((width, height), Image.BILINEAR)
    #lr = image.resize((1440, 720), Image.BILINEAR)
    #lr = image.resize((1800, 900), Image.BILINEAR)
    return lr

def main():
    path = './Cityscapes_train_HR'
    ratio = 6
    image_paths = [f.path for f in os.scandir(path) if f.is_file()]
    image_paths.sort()
    image_num = len(image_paths)
    start = time.time()
    for image_path in tqdm(image_paths):
        lr = down_sample(image_path, ratio)
        lr.save(os.path.join('./Cityscapes_train_LR_bicubic/X6', '{}'.format(os.path.basename(image_path))))
    print('time: {} s'.format(time.time()-start))
if __name__ == '__main__':
    main()
