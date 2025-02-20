import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import subprocess
import torch.backends.cudnn as cudnn
import torchmetrics
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import sys
import yaml
import os
import pandas as pd
os.chdir(sys.path[0])
sys.path.append(os.getcwd())
path_yaml = "./AGKD.yaml"
par = yaml.safe_load(open(path_yaml, 'r'))
sys.path.append('..')
import utils.par as Par
par = Par.Struct(**par)
root_dir = os.path.abspath(os.path.join(os.getcwd(),".."))  # xxx/AGKD
sys.path.append(root_dir)
torch.cuda.empty_cache()  # Clear CUDA cache

import datetime
from datetime import datetime, timedelta, timezone
from datasets.datasets import MyDataset, Data_embedding
from model.ResNet import MRAN
from tqdm import tqdm
from collections import defaultdict
from utils.func import get_bag_weight, mask_cross_entropy, weight_cross_entropy, ramp_up, compute_loss2
import numpy as np

cudnn.benchmark = True

utc_now = datetime.now()
beijing_now = utc_now.astimezone(timezone(timedelta(hours=8))).strftime('%m-%d-%H-%M')
print(beijing_now)
temp = str(par.distill_temp)

output_dir = par.output_dir if par.output_dir else time.strftime('%Y-%m-%d%H', time.localtime(time.time()))
output_dir = os.path.join(sys.path[0], 'run', output_dir, beijing_now,temp)

if not par.resume:
    with open(path_yaml,'r') as f:
        lines = f.readlines()
    with open(path_yaml,'w+') as f:
        if  'final_output_dir' not in par.__dict__:
            f.writelines(lines)
            f.write('final_output_dir: '+output_dir)
        else:
            lines = lines[:len(lines)-1]
            f.writelines(lines)
            f.write('final_output_dir: '+output_dir)
os.makedirs(output_dir, exist_ok=True)

cp_yaml_path = os.path.basename(path_yaml)
subprocess.call(f'cp {cp_yaml_path} {output_dir}', shell=True)

checkpoint_path = '/home/zhouyike/AGKD/Han/AGKD/AGKD/run/AGKD/02-04-12-14/3/checkpoint_epoch_1102-04-12-14.pth'
checkpoint = torch.load(checkpoint_path)

train_transforms = T.Compose([
    T.Resize((1024, 1024)),      
    T.RandomHorizontalFlip(),       
    T.ToTensor(),                 
])


val_transforms = T.Compose([
    T.Resize((1024, 1024)),        
    T.ToTensor(),                  
])

up_weight = defaultdict(lambda: 1.0)

wsi_alpha = {}
best_wsi_alpha = {}

bag_alpha = {}
bag_beta = {}
bag_label = {}

best_bag_alpha = {}
best_bag_bet = {}
best_bag_label = {}
best_acc_up = 0
best_acc_down = 0

final_recall = 0
final_precision = 0
final_sp = 0
final_f1 = 0
final_auc = 0

wsi_score = {}
wsi_alpha = {}

bag_score = {}

def get_path(save_dir):
    a_path = os.path.join(save_dir, 'alpha.pt')
    cp_path = os.path.join(save_dir, 'checkpoint.pth')
    config_path = os.path.join(save_dir, 'main.yaml')
    res_path = os.path.join(save_dir, 'result.pth')
    pred_path = os.path.join(save_dir, 'predict.csv')
    return a_path, cp_path, config_path, res_path, pred_path

alpha_path, _, config_path, res_path, pred_path = get_path(output_dir)

def evaluate(test_loader, model, epoch):
    num_wsi_test = test_loader.dataset.num_wsi
    embedding_test = [[] for _ in range(num_wsi_test)]
    label_test = [-1 for _ in range(num_wsi_test)]
    bag_id_test = [[] for _ in range(num_wsi_test)]

    correct = 0
    total = 0
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for i, (img, wsi_id, bag_id, label, _, bag_weight, _, _, patch_alpha) in enumerate(tqdm(test_loader, desc=f'val_{epoch}')):
            img = img.to('cuda:2')
            label = label.to('cuda:2')
            patch_alpha = patch_alpha.to('cuda:2')  # Make sure patch_alpha is passed and used
            
            with autocast():
                y, embedding, alpha, y_alpha, beta, y_beta = model(img, tag=0, patch_alpha=patch_alpha)
                loss = F.cross_entropy(y, label, reduction='mean')
                
            total_loss += loss.item()
            y_pred = y.argmax(dim=1)
            total += y_pred.size(0)
            correct += (y_pred == label).sum().item()

            for ind in range(len(wsi_id)):
                embedding_test[wsi_id[ind]].append(embedding[ind].tolist())
                label_test[wsi_id[ind]] = label[ind].item()
                bag_id_test[wsi_id[ind]].append(bag_id[ind].item())

    acc = correct / total

    return acc, total_loss, embedding_test, label_test, bag_id_test

def evaluate2(test_loader2, net, epoch,test=False):
    net.eval()

    correct = 0
    total = 0
    total_loss = 0

    recall_ = torchmetrics.Recall(average='none', num_classes=2)
    precision_ = torchmetrics.Precision(average='none', num_classes=2)
    auc_ = torchmetrics.AUROC(pos_label=1)
    recall_ = recall_.to('cuda:2')
    precision_ = precision_.to('cuda:2')
    auc_ = auc_.to('cuda:2')
    
    if test:
        pred_data = []

    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(tqdm(test_loader2, desc=f'val1_{epoch}')):
            embedding = embedding.to('cuda:2')
            label = label.to('cuda:2')
            y, alpha,_= net(embedding, tag=1)

            loss = F.cross_entropy(y, label)
            total_loss += loss.item()
            p = y.argmax()
            total += label.size(0)
            correct += (p == label).sum().item()

            auc_.update(y.softmax(dim=1)[:, 1], label)
            if test:
                pred_data.append([wsi_name[0], y.softmax(dim=1)[:, 1].item(), label.item()])
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)

    if test:
        pred = pd.DataFrame(columns=['wsi', 'p1', 'label'], data=pred_data)
        pred.to_csv(pred_path, index=False)
    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[1].item(), \
           auc_.compute().item(), recall_.compute()[0].item()

def get_dataloader(up_weight=None):
    pt = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_train.csv')
    pe = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_val.csv')

    train_data = MyDataset(path=pt, transform=train_transforms, up_weight=up_weight)
    val_data = MyDataset(path=pe, transform=val_transforms, up_weight=up_weight)

    train_loader = DataLoader(dataset=train_data, batch_size=par.batch_size, num_workers=par.num_workers,collate_fn=custom_collate_fn, drop_last=True, shuffle=True, pin_memory=True,prefetch_factor=8)
    val_bs = int(par.batch_size / 2)
    val_loader = DataLoader(dataset=val_data, batch_size=val_bs, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True,prefetch_factor=8)

    return train_loader, val_loader

def get_dataloader2(embedding, label, bag_id, wsi_name):
    data2 = Data_embedding(data=(embedding, label, bag_id, wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=False)

    return dataloader2

def custom_collate_fn(batch):
    """
    Custom collate_fn that handles variable-length sequences and aligns them to maximum length
    """
    # Find the maximum length in batch
    max_len = max([len(x[0]) for x in batch])  # If x[0] is a list, use len instead

    # Fill each sample so that they are the same size
    imgs = [F.pad(torch.tensor(x[0]), (0, 0, 0, max_len - len(x[0]))) for x in batch]
    labels = [x[1] for x in batch]  # Extract tags

    # Combine the processed data into batches
    imgs = torch.stack(imgs, dim=0)  # [batch_size, max_len]
    labels = torch.tensor([int(label) for label in labels])  # Make sure the label is an integer

    return imgs, labels

def test(model, best_pth=None):
    if best_pth is not None:
        model.load_state_dict(best_pth)  

    p_test = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_test.csv')

    test_data = MyDataset(p_test, transform=val_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True, prefetch_factor=8)

    up_acc, loss_test, embedding_test, label_test, bag_id_test = evaluate(test_loader, model, None)
    test_loader2 = get_dataloader2(embedding_test, label_test, bag_id_test, test_loader.dataset.wsi_name)
    acc, loss_test, precision, recall, auc, sp = evaluate2(test_loader2, model, None, True)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    torch.save(dict(acc=acc, auc=auc, f1=f1, se=recall, sp=sp), res_path)
    print(f"test_acc: {acc:.4f}  test_auc: {auc:.4f}   test_f1: {f1:.4f} test_precision: {precision:.4f}  test_se: {recall:.4f} test_sp: {sp:.4f}")
    return acc, auc, f1, recall, sp

def main():

    net = MRAN()
    if torch.cuda.is_available():
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        net.to(device) # Load the model onto GPU 2
        net.upstream = torch.nn.DataParallel(net.upstream, device_ids=[2,3]) 
        net.downstream = torch.nn.DataParallel(net.downstream, device_ids=[2,3])
         

    global best_acc_up, best_acc_down, bag_alpha, final_auc, final_recall, final_f1, final_precision

    # Extract best_pth from checkpoint
    best_pth = checkpoint['model_state_dict']  
    test(net, best_pth)
    torch.save(dict(best_bag_alpha=bag_alpha, best_bag_score=bag_score, best_bag_beta=bag_beta,
                    best_wsi_alpha=wsi_alpha, best_wsi_scrore=wsi_score, best_bag_label=bag_label), alpha_path)

# Example usage
if __name__ == "__main__":
    main()
