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
from train_mae import get_mean_std

class PixelCalculate():
    def __init__(self, args):
        mean, std = get_mean_std(args.dataset)
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([151.78], [48.85])]) # Samsung S7 ISP
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # Samsung S7 ISP
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
        # self.model = mae_vit_base_patch16(img_size=img_size, patch_size=(9,13), in_chans=1)
        self.model = torch.load('runs/train/MAE_ISP_0.7/exp6/weights/best.pt')
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.to(self.device)
        print('Using device:', self.device)
        self.model_path = os.path.join("runs/train", args.model_path)
        self.cls_thres = args.cls_thres
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
        # os.makedirs(self.weights_path)

        # self.logger = get_logger(os.path.join(self.save_path, f'{exp_dir}_train.log'))
        # self.logger.info(vars(args))


    def validate(self):
            self.model.eval()
            val_loss, normalize_term = 0, 0
            
            with torch.no_grad():
                with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                    for i, (org_img, feature, label, file) in enumerate(self.val_set):
                        org_img, feature, mask = org_img.to(self.device), feature.to(self.device), label.to(self.device)
                        loss, _, _ = self.model(feature, mask, org_img)
                        loss = loss.mean() # average on multi-gpu
                        val_loss += loss.item()             
                        pbar.set_postfix({'loss': loss.item()})
                        pbar.update()
                    pbar.close()
            val_loss /= len(self.val_set)
            
            return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['S7-ISP', 'FiveK'], default='FiveK')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cls_thres', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='5', help='GPU IDs to use, separated by commas. E.g., 0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--patch_num', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='/data1/Bad_Pixel_Detection/data/FiveK_Canon_0.7_0.01')
    parser.add_argument('--model_path', type=str, default='MAE_ISP_0.7')
    # parser.add_argument('--model_path', type=str, default='debug')
    args = parser.parse_args()
    print(args)
    pc = PixelCalculate(args)
    val_loss = pc.validate()
    print(val_loss)