import argparse
import os
import sys
import torch
from tqdm import tqdm
from dataset import SamsungDataset15x15
from torch.utils.data import DataLoader
from model.model import MLP_2L
from model.cnn import CNN
from model.mae_vit_single_layer import mae_vit_base_patch16
from utils.logger import get_logger
from utils.plot import plot_learning_curve
import math
from torchvision import transforms

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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = SamsungDataset15x15(args.data_path, 'train', transform=transform)
        val_data = SamsungDataset15x15(args.data_path, 'val', transform=transform)
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
    
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        self.device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.cluster_size = args.cluster_size
        self.val_step = args.val_step
        # self.model = torch.nn.DataParallel(CNN(self.cluster_size).to(self.device), device_ids=[args.device])
        self.model = mae_vit_base_patch16(img_size=(15,15), in_chans=1).to(self.device)
        if args.pretrained:
            self.model = torch.load(args.pretrained)
        self.model.to(self.device)
        self.model_path = args.model_path
        self.criterion = torch.nn.MSELoss()
        self.dataset = args.dataset
        idx = 0
        exp_dir = 'exp'
        if not os.path.exists(self.model_path):
            self.save_path = os.path.join(self.model_path, 'exp')
        else:
            while(os.path.exists(os.path.join(self.model_path, exp_dir))):
                idx += 1
                exp_dir = f'exp{idx}'
            self.save_path = os.path.join(self.model_path, exp_dir)
        self.save_path = os.path.join(self.save_path, 'train')
        self.weights_path = os.path.join(self.save_path, 'weights')
        os.makedirs(self.weights_path)

        self.logger = get_logger(os.path.join(self.save_path, f'{exp_dir}_train.log'))
        self.logger.info(vars(args))
        self.logger.info(f"Reading training data from {train_path}, validation data from {val_path}.")
        
        
    def save_model(self, save_path, file_name):
        f = os.path.join(save_path, file_name + ".pt")
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module, f)
        else:
            torch.save(self.model, f)


    def validate(self):
        self.model.eval()
        val_loss, normalize_term = 0, 0
        val_mse, val_nmse = 0, 0
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.val_set):
                    feature, label = feature.to(torch.float32).to(self.device), label.to(torch.float32).to(self.device)
                    # feature = feature.unsqueeze(1) # add channel dimension
                    predict = self.model(feature)
                    mean, std = get_mean_std(self.dataset)
                    predict = predict*std[0] + mean[0]   
                    loss = self.criterion(predict, label)
                    normalize_term += torch.sum(torch.pow(label, 2)).item() / torch.numel(label)
                    val_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
        val_nmse = val_loss/normalize_term
        val_mse = val_loss/len(self.val_set)
        return val_nmse, val_mse


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        
        train_loss, val_loss, normalize_term, min_val_mse, min_val_nmse = 0, 0, 0, sys.maxsize, sys.maxsize
        train_mse, train_nmse = 0, 0
        loss_vec, val_loss_vec, val_vec = [], [], []
        model.train()
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.train_set):
                    # adjust_learning_rate(optimizer, i / len(self.train_set) + epoch, args)
                    feature, label = feature.to(torch.float32).to(self.device), label.to(torch.float32).to(self.device)
                    # feature = feature.unsqueeze(1) # add channel dimension
                    optimizer.zero_grad()
                    predict = model(feature) 
                    mean, std = get_mean_std(self.dataset)    
                    predict = predict*std[0] + mean[0]   
                    loss = self.criterion(predict, label)
                    normalize_term += torch.sum(torch.pow(label, 2)).item() / torch.numel(label)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
                
            # train_loss /= normalize_term
            train_mse = train_loss/len(self.train_set)
            train_nmse = train_loss/normalize_term
            loss_vec.append(train_nmse)
            info = f"Epoch: {epoch + 1}\tTraining NMSE: {train_nmse}\tTraining MSE: {train_mse}"
            if (epoch + 1) % self.val_step:
                self.logger.info(info)
            else:
                val_nmse, val_mse = self.validate()
                info += f"\tValidation NMSE: {val_nmse}\tValidation MSE: {val_mse}"
                self.logger.info(info)
                val_vec.append(epoch + 1)
                val_loss_vec.append(val_nmse)

                self.save_model(self.weights_path, 'last')
                if val_nmse < min_val_nmse:
                    # self.lr /= 2
                    min_val_nmse = val_nmse
                    self.save_model(self.weights_path, 'best')
                if val_mse < min_val_mse:
                    min_val_mse = val_mse
            
        self.logger.info("Training Completed.")
        self.logger.info(f"Best validation NMSE: {min_val_nmse}")
        self.logger.info(f"Best validation MSE: {min_val_mse}")
        # psnr = 20 * log10(1024 / sqrt(min_val_loss))
        # self.logger.info(f"Best validation PSNR: {psnr}")

        plot_learning_curve(loss_vec, val_vec, val_loss_vec, self.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--device', type=int, nargs='+', default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--use_poison', action='store_true', help='train on patches with corrupted pixels in neighborhood')
    parser.add_argument('--data_path', type=str, default='/data1/Bad_Pixel_Correction/FiveK/feature_15')
    parser.add_argument('--model_path', type=str, default='results/cnn')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'vit'])
    parser.add_argument('--cluster_size', type=int, default=5)
    parser.add_argument('--dataset', type=str, choices=['S7-ISP', 'FiveK'], default='FiveK')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained', type=str, default='')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    if args.eval:
        val_nmse, val_mse = pc.validate()
        print(f"Validation NMSE: {val_nmse}\tValidation MSE: {val_mse}")
    else:
        pc.train()