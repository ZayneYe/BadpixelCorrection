import argparse
import os
import sys
import torch
from tqdm import tqdm
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model.mae_vit import mae_vit_base_patch16
from utils.logger import get_logger
# from utils.plot import plot_loss_curve, plot_prcurve, plot_lr_curve, plot_iou_curve
# from utils.metrics import calc_metrics, dice_loss, calc_metrics_one
# from utils.process import postprocess, generate_pred_dict
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math 
import torch.nn as nn

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    args.min_lr = 0
    args.warmup_epochs = 5
    if epoch < args.warmup_epochs: # warmup_epochs
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def get_mean_std(dataset):
    mean = {'S7-ISP': [151.78], 'FiveK': [491.60]}
    std = {'S7-ISP': [48.85], 'FiveK': [571.15]}
    return mean[dataset], std[dataset]

class PixelCalculate():
    def __init__(self, args):
        mean, std = get_mean_std(args.dataset)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # Samsung S7 ISP
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([491.60], [571.15])]) # MIT FiveK
        # transform = transforms.Compose([transforms.ToTensor()])
        mask_transform = transforms.Compose([transforms.ToTensor()])
        train_data = SamsungDataset(args.data_path, cate='train', transform=transform, mask_transform=mask_transform, patch_num=args.patch_num, dataset=args.dataset)
        val_data = SamsungDataset(args.data_path, cate='val', transform=transform, mask_transform=mask_transform, patch_num=args.patch_num, dataset=args.dataset)
            
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        self.dataset = args.data_path.split("/")[1][:3]
        # self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        device_ids = [int(id) for id in args.device.split(',')]
        self.device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.val_step = args.val_step
        img_size = train_data[0][0].shape[1:]
        input_channel = train_data[0][0].shape[0]
        # self.model = mae_vit_base_patch16(img_size=img_size, patch_size=(9,12), in_chans=1)
        patch_size = (9,12) if args.dataset == 'S7-ISP' else (9,13)
        self.model = mae_vit_base_patch16(img_size=img_size, patch_size=patch_size, in_chans=1)
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.to(self.device)
        print('Using device:', self.device)
        self.model_path = os.path.join("runs/train", args.model_path)
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        idx = 0
        exp_dir = 'exp'
        if not os.path.exists(self.model_path):
            self.save_path = os.path.join(self.model_path, 'exp')
        else:
            while(os.path.exists(os.path.join(self.model_path, exp_dir))):
                idx += 1
                exp_dir = f'exp{idx}'
            self.save_path = os.path.join(self.model_path, exp_dir)
        # self.save_path = os.path.join(self.save_path, 'train')
        self.weights_path = os.path.join(self.save_path, 'weights')
        os.makedirs(self.weights_path)

        self.logger = get_logger(os.path.join(self.save_path, f'{exp_dir}_train.log'))
        self.logger.info(vars(args))
        
        
    def save_model(self, save_path, file_name):
        f = os.path.join(save_path, file_name + ".pt")
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module, f)
        else:
            torch.save(self.model, f)


    def validate(self):
        self.model.eval()
        val_loss, normalize_term = 0, 0
        
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                for i, (org_img, feature, label, file) in enumerate(self.val_set):
                    org_img, feature, mask = org_img.to(self.device), feature.to(self.device), label.to(self.device)
                    loss, _, _ = self.model(feature, mask, org_img)
                    loss = loss.mean() # average on multi-gpu
                    # if not math.isnan(loss): # error inject failed for few images, so mask for them is all zeros
                    val_loss += loss.item()             
                    # pred_dict = generate_pred_dict(pred_dict, file, predict, label)
                    # loss = self.criterion(predict, label)
                    # loss += dice_loss(predict, label)
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
        val_loss /= len(self.val_set)
      
        # pred_all, lab_all = postprocess(pred_dict, self.dataset)
        # if self.dataset == "ISP":
        #     recall, precision, iou, dice_score, _ = calc_metrics(pred_all, lab_all, thres)
        # else:
        #     recall, precision, iou, dice_score = calc_metrics_one(pred_all, lab_all, thres)
        
        return val_loss


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001) # optimizer for pixel correction
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
        
        train_loss, val_loss, min_val_loss = 0, 0, 10000
        loss_vec, val_loss_vec, val_vec, lr_vec, iou_score_vec, dice_score_vec = [], [], [], [], [], []
        model.train()
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
                for i, (org_img, feature, label, _) in enumerate(self.train_set):
                    adjust_learning_rate(optimizer, i / len(self.train_set) + epoch, args)
                    org_img, feature, label = org_img.to(self.device), feature.to(self.device), label.to(self.device)
                    
                    optimizer.zero_grad()
                    loss, _, _ = model(feature, label, org_img)
                    # loss = self.criterion(predict, label)
                    # loss += dice_loss(predict, label)
                    # cm, r, p, iou, tn, fp, fn, tp = calc_metrics(predict, label, self.cls_thres)
                    loss = loss.mean() # average on multi-gpu
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    lr_vec.append(optimizer.param_groups[0]["lr"])
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
            # scheduler.step()
            train_loss /= len(self.train_set)
            loss_vec.append(train_loss)
            info = f"Epoch: {epoch + 1}\tTraining Loss: {train_loss:.4f}\tlr: {optimizer.param_groups[0]['lr']:.6f}\t"
            if (epoch + 1) % self.val_step:
                self.logger.info(info)
            else:
                val_loss = self.validate()
                info += f"Validation Loss: {val_loss:.4f}"
                self.logger.info(info)
                val_vec.append(epoch + 1)
                val_loss_vec.append(val_loss)
             
                self.save_model(self.weights_path, 'last')
                if val_loss < min_val_loss:
                    # self.lr /= 2
                    min_val_loss = val_loss
                    self.save_model(self.weights_path, 'best')
            # if epoch + 1 == self.epochs:
            #     r_vec, p_vec, iou_vec, dice_vec = [], [], [], []
            #     for c in np.linspace(0, 1, 31):
            #         _, r, p, iou, dice = self.validate(c)
            #         # print(r, p, c)
            #         r_vec.append(r)
            #         p_vec.append(p)
            #         iou_vec.append(iou)
            #         dice_vec.append(dice)
                
            
        self.logger.info("Training Completed.")
     
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['S7-ISP', 'FiveK'], default='S7-ISP')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='3,5', help='GPU IDs to use, separated by commas. E.g., 0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--patch_num', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='/data1/Bad_Pixel_Detection/data/ISP_0.7_0.7')
    parser.add_argument('--model_path', type=str, default='MAE_ISP_0.7')
    # parser.add_argument('--model_path', type=str, default='debug')
    args = parser.parse_args()
    print(args)
    pc = PixelCalculate(args)
    pc.train()