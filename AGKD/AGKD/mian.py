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
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR

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
import copy

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

if not par.resume:
    cp_path = os.path.join(output_dir, 'checkpoint.pth')
else:
    cp_path = os.path.join(par.final_output_dir, 'checkpoint.pth')
if not par.resume and os.path.exists(cp_path):
    os.remove(cp_path)

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

def train(train_loader, epoch, model, optimizer, scheduler=None, soft_labels=None):
    scaler = torch.cuda.amp.GradScaler()
    hard_lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    soft_lossfunc = nn.KLDivLoss(reduction='batchmean')
    hard_lossfunc = hard_lossfunc.to('cuda:4')
    soft_lossfunc = soft_lossfunc.to('cuda:4')

    num_wsi_train = train_loader.dataset.num_wsi
    temp_embedding = [[] for _ in range(num_wsi_train)]
    temp_label = [-1 for _ in range(num_wsi_train)]
    temp_bag_id = [[] for _ in range(num_wsi_train)]

    total_loss = 0
    soft_loss_total = 0
    t0, t1, t2 = 0, 0, 0

    ramp_up1 = par.Lambda * ramp_up(epoch, par.num_epoch)
    ramp_up2 = par.Lambda ** 2 * ramp_up(epoch, par.num_epoch)

    cnt_one = 0
    start_time = time.time()
    model.train()

    for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, wsi_name, index, patch_alpha) in enumerate(tqdm(train_loader, desc=f'train_{epoch}')):
        img = img.to('cuda:4')
        patch_alpha = patch_alpha.to('cuda:4')  # Make sure patch_alpha is passed and used

        if soft_labels is not None:
            fake_label = []
            bag_id = bag_id.tolist()
            for i, k in enumerate(wsi_name):
                index = list(zip(*np.where(np.array(soft_labels, dtype=object) == k)))[0]
                soft_bag_id = soft_labels[index[0]][1][0]
                if bag_id[i] in soft_bag_id:
                    index_2 = list(zip(*np.where(np.array(soft_bag_id, dtype=object) == bag_id[i])))[0]
                    fake_label.append(torch.from_numpy(np.array(soft_labels[index[0]][0][index_2[0]])))
                else:
                    fake_label.append(torch.from_numpy(np.random.uniform(-1, 1, 2)))
            fake_label = torch.stack(fake_label).to('cuda:4')
            fake_label = fake_label.to(torch.float32)
            label = label.to('cuda:4')
        else:
            label = label.to('cuda:4')

        cnt_one += (bag_weight == 1).sum().item()
        bag_weight = bag_weight.to('cuda:4')

        with autocast():
            y, embedding, alpha, y_alpha, beta, y_beta = model(img, tag=0, patch_alpha=patch_alpha)

            loss0 = mask_cross_entropy(y, label, bag_weight, loss=hard_lossfunc)
            w1 = bag_weight.unsqueeze(dim=1) * alpha
            loss1 = 0*weight_cross_entropy(y_alpha.view(-1, 2), label.repeat(64, 1).T.contiguous().view(-1), w1.view(-1), loss=hard_lossfunc)
            w2 = (bag_weight.unsqueeze(dim=1) * alpha).view(-1).unsqueeze(dim=1)
            w2 = (w2 * beta).view(-1)

            loss2 = 0*weight_cross_entropy(y_beta.view(-1, 2), label.repeat(64 * 16, 1).T.contiguous().view(-1), w2, loss=hard_lossfunc)

            loss = loss0 + ramp_up1 * loss1 + ramp_up2 * loss2 
            t0 += loss0.item()
            t1 += ramp_up1 * loss1.item()
            t2 += ramp_up2 * loss2.item()

            if soft_labels is not None:
                T = par.distill_temp
                soft_loss = soft_lossfunc(F.log_softmax(y/T, dim=1), F.softmax(fake_label/T, dim=1))
                distill_alpha = 5
                loss += distill_alpha * soft_loss
                soft_loss_total += soft_loss.item()

            total_loss += loss.item()

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        for ind in range(len(index)):
            temp_embedding[wsi_id[ind]].append(embedding[ind].tolist())
            temp_label[wsi_id[ind]] = label[ind].item()
            if soft_labels is not None:
                temp_bag_id[wsi_id[ind]].append(bag_id[ind])
            else:
                temp_bag_id[wsi_id[ind]].append(bag_id[ind].item())

    if soft_labels is None:
        print(f"hard_loss: {t0:.4f} total_loss: {total_loss:.4f}")
    else:
        print(f"hard_loss: {t0:.4f} total_loss: {total_loss:.4f} soft_loss: {soft_loss_total:.4f}")

    return total_loss, temp_embedding, temp_label, temp_bag_id, cnt_one

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
            img = img.to('cuda:4')
            label = label.to('cuda:4')
            patch_alpha = patch_alpha.to('cuda:4')  # Make sure patch_alpha is passed and used
            
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

def train2(train_loader, epoch, net, optimizer, scheduler=None):
    total_loss = 0
    t0 = 0
    t1 = 0
    t2 = 0

    lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    lossfunc.to('cuda:4')

    ramp_up3 = par.Lambda * ramp_up(epoch, par.num_epoch)

    up_weight.clear()
    net.train()
    bag_soft = []

    start_time = time.time()
    for i, (embedding, label, bag_id, wsi_name, index) in enumerate(tqdm(train_loader, desc=f'train1_{epoch}')):
        embedding = embedding.to('cuda:4')
        label = label.to('cuda:4')

        y, alpha, y_alpha = net(embedding, tag=1)
        bag_soft.append([y_alpha.tolist(),bag_id.tolist(),wsi_name[0]])

        loss0 = F.cross_entropy(y, label) if not lossfunc else lossfunc(y, label)
        loss1 = weight_cross_entropy(y_alpha, label.repeat(alpha.shape[0]), alpha, loss=lossfunc)
        loss2 = compute_loss2(alpha,y_alpha)


        loss = loss0 + ramp_up3 * loss1 + 5 * loss2
        t0 += loss0.item()
        t1 += ramp_up3 * loss1.item()
        t2 += 5 * loss2.item()

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        temp = get_bag_weight(wsi_name[0], bag_id.squeeze(dim=0).tolist(), alpha.tolist())
        up_weight.update(temp)

    print(f"_loss0: {t0:.4f}  _loss1: {t1:.4f} _loss2: {t2:.4f}  total_loss: {total_loss:.4f}")

    scheduler.step()

    return bag_soft,total_loss

def evaluate2(test_loader2, net, epoch,test=False):
    net.eval()

    correct = 0
    total = 0
    total_loss = 0

    recall_ = torchmetrics.Recall(average='none', num_classes=2)
    precision_ = torchmetrics.Precision(average='none', num_classes=2)
    auc_ = torchmetrics.AUROC(pos_label=1)
    recall_ = recall_.to('cuda:4')
    precision_ = precision_.to('cuda:4')
    auc_ = auc_.to('cuda:4')
    
    if test:
        pred_data = []

    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(tqdm(test_loader2, desc=f'val1_{epoch}')):
            embedding = embedding.to('cuda:4')
            label = label.to('cuda:4')
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

    train_loader = DataLoader(dataset=train_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=True, pin_memory=True,prefetch_factor=8)
    val_bs = int(par.batch_size / 2)
    val_loader = DataLoader(dataset=val_data, batch_size=val_bs, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True,prefetch_factor=8)

    return train_loader, val_loader

def get_dataloader2(embedding, label, bag_id, wsi_name):
    # Filter out empty or abnormal samples
    valid_indices = [i for i in range(len(embedding)) if len(embedding[i]) > 0]
    
    if len(valid_indices) == 0:
        print("Warning: All samples are empty or invalid in the dataloader!")
        return None  
    
    filtered_embedding = [embedding[i] for i in valid_indices]
    filtered_label = [label[i] for i in valid_indices]
    filtered_bag_id = [bag_id[i] for i in valid_indices]
    filtered_wsi_name = [wsi_name[i] for i in valid_indices]

    data2 = Data_embedding(data=(filtered_embedding, filtered_label, filtered_bag_id, filtered_wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=True)
    
    return dataloader2


def downstream_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch_up):

    id_up = f'{epoch_up:0{2}}'
    bag_soft, loss= train2(train_loader2, epoch_up, net, optimizer2, scheduler2)
    acc, loss_val, precision, recall, auc, sp = evaluate2(val_loader2, net, epoch=epoch_up)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f'_epoch_{id_up}:  train loss: {loss:.4f}      val loss: {loss_val:.4f}      val acc: {acc:.6f} '
          f'auc: {auc:.6f}   f1: {f1:.6f}    se: {recall:.6f}       sp: {sp:.6f}   precision: {precision:.6f}')

    return acc, recall, precision, sp, f1, auc, bag_soft

def test(model, best_pth=None):
    if best_pth is not None:
        model.load_state_dict(best_pth)  # Use the best model parameters

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
    soft = beijing_now + '-AGKD' + str(par.distill_temp)
    rootfile_dir = os.path.abspath(os.getcwd())
    result_folder_path = os.path.join(rootfile_dir, par.result_dir, soft)
    os.makedirs(result_folder_path, exist_ok=True)

    tea_res_path = os.path.join(result_folder_path, beijing_now + 'teacher_result.txt')
    stu_res_path = os.path.join(result_folder_path, beijing_now + '_student_result.txt')
    os.mknod(tea_res_path)
    os.mknod(stu_res_path)

    global up_weight

    train_loader, val_loader = get_dataloader()  # Assume it gets all necessary parameters itself

    net = MRAN()
    
    # Count parameters
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # total_params_in_bytes = total_params * 4  
    # total_params_in_MB = total_params_in_bytes / (1024 ** 2)  # Convert to MB
    # print(f"Total number of parameters: {total_params_in_MB:.2f} MB")
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        net.to(device)  # Load the model onto GPU 4
        net.upstream = torch.nn.DataParallel(net.upstream, device_ids=[4,5]) 
        net.downstream = torch.nn.DataParallel(net.downstream, device_ids=[4,5]) 
        
    with open(tea_res_path, "a") as myfile:
        myfile.write("epoch\t\t\tloss\t\t\tacc\t\t_acc\t\trecall\t\tprecision\t\tsp\t\tf1\t\tauc\n")
    with open(stu_res_path, "a") as myfile:
        myfile.write("epoch\t\t\tloss\t\t\tacc\t\t_acc\t\trecall\t\tprecision\t\tsp\t\tf1\t\tauc\n")

    optimizer1 = torch.optim.Adam(net.upstream.parameters(), lr=par.learning_rate1)
    scheduler1 = OneCycleLR(optimizer1, max_lr=par.learning_rate1, steps_per_epoch= 6*int(len(train_loader)), epochs=par.num_epoch)
    optimizer2 = torch.optim.Adam(net.downstream.parameters(), lr=par.learning_rate2)
    scheduler2 = MultiStepLR(optimizer2, milestones=[5], gamma=.5)

    global best_acc_up, best_acc_down, bag_alpha, final_auc, final_recall, final_f1, final_precision
    best_acc_down_epoch = -1

    start_epoch = 0

    if par.resume and os.path.exists(cp_path):
        checkpoint = torch.load(cp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
        scheduler1.load_state_dict(checkpoint['scheduler1_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        scheduler2.load_state_dict(checkpoint['scheduler2_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loader, val_loader = get_dataloader()  # Reload dataloaders if necessary

    for epoch in range(start_epoch, par.num_epoch):

        loss, temp_embedding, temp_label, temp_bag_id,_ = train(train_loader, epoch, net, optimizer1, scheduler1)
        #print(temp_embedding, temp_label.shape)

        # Train for soft labels
        acc, loss_val, embedding_val, label_val, bag_id_val = evaluate(val_loader, net, epoch=epoch)
        print(f'teacher_epoch: {epoch} teacher_train loss: {loss:.2f}   teacher_val loss: {loss_val:.2f}   teacher_val acc: {acc:.6f}')

        train_loader2 = get_dataloader2(temp_embedding, temp_label, temp_bag_id, train_loader.dataset.wsi_name)
        val_loader2 = get_dataloader2(embedding_val, label_val, bag_id_val, val_loader.dataset.wsi_name)

        _acc, recall, precision, sp, f1, auc, bag_soft= downstream_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch)
        with open(tea_res_path, "a") as myfile:
            myfile.write(f"{epoch}\t\t\t{loss:.4f}\t\t\t{acc:.4f}\t\t{_acc:.4f}\t\t{recall:.4f}\t\t{precision:.4f}\t\t{sp:.4f}\t\t{f1:.4f}\t\t{auc:.4f}\n")

        # Use soft-labels for distillation
        loss, temp_embedding, temp_label, temp_bag_id,_ = train(train_loader, epoch, net, optimizer1, scheduler1,bag_soft)
        acc, loss_val, embedding_val, label_val, bag_id_val = evaluate(val_loader, net, epoch=epoch)
        print(f'student_epoch:{epoch}  student_train loss: {loss:.2f}   student_val loss: {loss_val:.2f}   student_val acc: {acc:.6f}')

        train_loader2 = get_dataloader2(temp_embedding, temp_label, temp_bag_id, train_loader.dataset.wsi_name)
        val_loader2 = get_dataloader2(embedding_val, label_val, bag_id_val, val_loader.dataset.wsi_name)
        _acc, recall, precision, sp, f1, auc, _ = downstream_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch)
        if len(train_loader2.dataset) == 0:
            raise ValueError("Generated train_loader2 is empty, check embedding and labels.")
        
        if train_loader2 is None or len(train_loader2.dataset) == 0:
            print(f"Epoch {epoch}: Skipping downstream task due to empty train_loader2 after distillation.")
            continue
        
        with open(stu_res_path, "a") as myfile:
            myfile.write(f"{epoch}\t\t\t{loss:.4f}\t\t\t{acc:.4f}\t\t{_acc:.4f}\t\t{recall:.4f}\t\t{precision:.4f}\t\t{sp:.4f}\t\t{f1:.4f}\t\t{auc:.4f}\n")

        if acc > best_acc_up:
            best_acc_up = acc
            
        if _acc >= best_acc_down:
            best_acc_down = _acc
            best_acc_down_epoch = epoch
            final_recall = recall
            final_precision = precision
            final_sp = sp
            final_f1 = f1
            final_auc = auc
            best_pth = copy.deepcopy(net.state_dict())
            
        
        # Select a Save checkpoint
        # if _acc >= 0.8:
        #     checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}{beijing_now}.pth')
        #     save_epoch=20
        #     torch.save({
        #         'epoch': save_epoch,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer1_state_dict': optimizer1.state_dict(),
        #         'scheduler1_state_dict': scheduler1.state_dict(),
        #         'optimizer2_state_dict': optimizer2.state_dict(),
        #         'scheduler2_state_dict': scheduler2.state_dict(),
        #         'up_weight': dict(up_weight),
        #         'best_pth': best_pth
        #     }, checkpoint_path)
        
        
        # Save the best checkpoint
        torch.save({'id': id, 'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    'scheduler1_state_dict': scheduler1.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'scheduler2_state_dict': scheduler2.state_dict(),
                    'up_weight': dict(up_weight),
                    'best_pth': best_pth}, cp_path)

        train_loader, val_loader = get_dataloader(up_weight)


    print(f'best acc_up: {best_acc_up:.4f}     best acc_down: {best_acc_down:.4f}   final_f1: {final_f1:.4f}   '
          f'final_auc: {final_auc:.4f}    best_acc_epoch:{best_acc_down_epoch:.4f}')

    print(f'training completed.\n checkpoint is saved in output_dir: {output_dir}') 

    # Instant test results after training
    test(net, best_pth)# Test directly with the currently trained model
    torch.save(dict(best_bag_alpha=bag_alpha, best_bag_score=bag_score, best_bag_beta=bag_beta,
                    best_wsi_alpha=wsi_alpha, best_wsi_scrore=wsi_score, best_bag_label=bag_label), alpha_path)

# Example usage
if __name__ == "__main__":
    main()
