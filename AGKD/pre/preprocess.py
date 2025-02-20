import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import slideio
from PIL import Image
import subprocess
from collections import Counter
import pandas as pd
from multiprocessing import Pool
import glob
import argparse
import openslide
import time
import torchvision.transforms as T
root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # xxx/AGKD

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--save_dir', type=str, default='/AGKD/camelyon16-img', help='directory to save csv file')
parser.add_argument('--dataset_dir', type=str, default='/AGKD/camelyon16', help='svs datasets storage directory')

args = parser.parse_args()
img_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
])
def cut_wsi2(wsi_path):
    b = 2048
    save_loc = True

    # save location of sub pic rather than sub pic
    name = wsi_path.split('/')[-1][:-4]
    d = os.path.join(args.save_dir,'loc', f'{name}.csv')
    if save_loc and os.path.exists(d):
        print(f'{d} exists')
        return

        # b = 8 * 512
    slide = openslide.OpenSlide(wsi_path)

    [w, h] = slide.level_dimensions[0]

    m, n = h // b, w // b  # m x n  bag

    sel = [0] * m * n
    cnt_bag = 0

    col = ['bag_id', 'x', 'y']
    data = []
    
    save_img_dir = os.path.join(args.save_dir,'save_img', name)
    os.makedirs(save_img_dir, exist_ok=True)

    for i in range(m):  # traverse all bag
        start = time.time()
        for j in range(n):
            lu = [i * b, j * b]  # left_up (h,w)
            ld = [i * b + b - 1, j * b]
            ru = [i * b, j * b + b - 1]
            rd = [i * b + b - 1, j * b + b - 1]
            bag_loc = [lu, ld, ru, rd]
            bag_id = i * n + j

            bag_image = np.array(slide.read_region((lu[1], lu[0]),0, (b, b)).convert('RGB'))  # image: H x W x 3
            ret, mask = cv2.threshold(bag_image, 127, 255, cv2.THRESH_BINARY)

            temp = np.all(mask == 255, axis=2)
            sel[bag_id] = temp.sum() / temp.size

            if sel[bag_id] < 0.98:
                data.append([bag_id, lu[1], lu[0]])
                save_img = img_transforms(bag_image)
                save_img.save(os.path.join(save_img_dir, f'{bag_id}_{lu[1]}_{lu[0]}.png'))

    end = time.time()
    print(f'{wsi_path}','time:',end-start)

    t = pd.DataFrame(columns=col, data=data)
    t.to_csv(d, index=False)
    return sel, m, n

if __name__ == '__main__':

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'loc'), exist_ok=True)
    li_abspath = glob.glob(f'{args.dataset_dir}/*/*/*.svs')  # change the '.svs' when you try to use another kind of wsi

    pool = Pool(24)
    pool.map(cut_wsi2, li_abspath)
    pool.close()
    pool.join()
