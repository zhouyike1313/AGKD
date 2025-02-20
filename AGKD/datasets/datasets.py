import glob
import os
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from multiprocessing import Process
import multiprocessing
from PIL import Image
from collections import defaultdict, Counter
import slideio
from tqdm import tqdm, trange
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transform=None, up_weight=None, len_bagid=4, bag_size=2048, patch_alpha=None):
        t = pd.read_csv(path)
        self.bag_size = bag_size
        #use_loc is the judgment is segmented
        self.use_loc = 1 if 'x' in list(t) else 0
        if not self.use_loc:
            self.label = list(t['type'])
            self.li_path = list(t['path'])
            li_wsi = list(t['wsi_name'])
            self.wsi_name = list(set(t['wsi_name']))
            self.num_wsi = len(self.wsi_name)
            d = {i: j for i, j in zip(self.wsi_name, range(self.num_wsi))}
            self.wsi_id = [d[i] for i in li_wsi]  # todo check
            self.num_bag = len(self.li_path)
            self.transform = transform

            self.bag_id = [int(i[-(4 + len_bagid):-4]) for i in self.li_path]

            self.up_weight = up_weight if up_weight is not None else defaultdict(lambda: 1.0)

            # todo class weight by bag ,or class weight by wsi
            # counter = dict(Counter(self.label))
            s1 = set([(i, j) for i, j in zip(li_wsi, self.label)])
            counter = Counter([i[1] for i in s1])

            self.loss_weight = torch.Tensor([counter[1], counter[0]])
            self.loss_weight /= self.loss_weight.sum()
        else:
            # Read file contents directly
            self.x = list(t['x'])
            self.y = list(t['y'])
            self.label = list(t['type'])
            self.bag_id = list(t['bag_id'])
            self.wsi_path = list(t['wsi'])
            self.li_wsi_name = [i.split('/')[-1][:-4] for i in self.wsi_path]
            self.li_path = [i + '_' + str(j) for i, j in zip(self.li_wsi_name, self.bag_id)]
            self.wsi_name = list(set(self.li_wsi_name))
            self.num_wsi = len(self.wsi_name)
            d = {i: j for i, j in zip(self.wsi_name, range(self.num_wsi))}
            self.wsi_id = [d[i] for i in self.li_wsi_name]
            self.num_bag = len(self.label)
            self.transform = transform
            self.up_weight = up_weight if up_weight is not None else defaultdict(lambda: 1.0)
            
            s1 = set([(i, j) for i, j in zip(self.wsi_path, self.label)])
            counter = Counter([i[1] for i in s1])

            self.loss_weight = torch.Tensor([counter[1], counter[0]])
            self.loss_weight /= self.loss_weight.sum()
        self.patch_alpha = patch_alpha

    def __getitem__(self, index):
        if not self.use_loc:
            img = Image.open(self.li_path[index])
        else:
            img_id_x_y = f"{self.bag_id[index]}_{self.x[index]}_{self.y[index]}.png"
            img_dir = os.path.join('/home/zhouyike/AGKD/Han/AGKD/csv-camelyon16/example/loc/save_img', self.wsi_path[index].split('/')[-1][:-4], img_id_x_y)
            img = Image.open(img_dir)
        if self.transform is not None:
            # If the conversion process requires converting an image from PIL to Tensor, make sure ToPILImage() is only applied when necessary
            # Check whether the transform already contains ToPILImage. If it does, and the input is already PIL.Image, then ToPILImage is skipped
            if any(isinstance(t, transforms.ToPILImage) for t in self.transform.transforms):
                transformed_img = []
                for t in self.transform.transforms:
                    if isinstance(t, transforms.ToPILImage) and isinstance(img, Image.Image):
                        continue  # If the image is already PIL type, skip the ToPILImage conversion
                    img = t(img)
                img = img
            else:
                img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.patch_alpha is None:
            patch_alpha_i = -1
        else:
            key_patch_alpha = self.wsi_path[index].split('/')[-1][:-4] + '_' + str(self.bag_id[index])
            patch_alpha_i = torch.tensor(self.patch_alpha[key_patch_alpha])

        return img.type(dtype=torch.float16), self.wsi_id[index], self.bag_id[index], self.label[index], self.li_path[index], torch.tensor(self.up_weight[self.li_path[index]]), self.li_wsi_name[index], index, patch_alpha_i


    def __len__(self):
        return self.num_bag


class Data_embedding(torch.utils.data.Dataset):
    def __init__(self, path=None, data=None):
        if data is not None:
            self.embedding, self.label, self.bag_id, self.wsi_name = data
        else:
            data = torch.load(path)
            self.embedding = data['embedding']
            self.label = data['label']
            self.bag_id = data['bag_id']
            self.wsi_name = data['wsi_name']

        self.len = len(self.label)

        counter = Counter(self.label)
        self.loss_weight = torch.Tensor([counter[1], counter[0]])
        self.loss_weight /= self.loss_weight.sum()

        for i in range(len(self.embedding)):
            self.embedding[i] = torch.tensor(self.embedding[i])

    def __getitem__(self, index):
        return self.embedding[index], self.label[index], torch.tensor(self.bag_id[index]), \
               self.wsi_name[index], index

    def __len__(self):
        return self.len





