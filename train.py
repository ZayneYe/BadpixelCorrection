import argparse
import os
import sys
import torch
from tqdm import tqdm
from dataset import SamsungDatasetSmall
from torch.utils.data import DataLoader
from model.model import MLP_2L
from utils.logger import get_logger
from utils.plot import plot_learning_curve
from math import log10, sqrt


class PixelCalculate():
    def __init__(self, args):
        if args.use_poison:
            train_data = SamsungDatasetSmall(args.data_path, 'poison_train_0.7')
            val_data = SamsungDatasetSmall(args.data_path, 'poison_val_0.7')
            train_path = os.path.join(args.data_path, 'poison_train_0.7')
            val_path = os.path.join(args.data_path, 'poison_val_0.7')
        else:
            train_data = SamsungDatasetSmall(args.data_path, 'train')
            val_data = SamsungDatasetSmall(args.data_path, 'val')
            train_path = os.path.join(args.data_path, 'train')
            val_path = os.path.join(args.data_path, 'val')
        
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        self.device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.patch_size = len(train_data[0][0])
        self.val_step = args.val_step
        self.model = torch.nn.DataParallel(MLP_2L(self.patch_size).to(self.device), device_ids=[args.device])
        self.model_path = args.model_path
    
        self.criterion = torch.nn.MSELoss()

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
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.val_set):
                    input, target = feature.to(torch.float32).to(self.device), label.to(torch.float32).to(self.device)
                    predict = self.model(input)
                    predict = predict.view(len(predict))
                    loss = self.criterion(predict, target)
                    normalize_term += sum(pow(target, 2)).item() / len(target)
                    val_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
        val_loss /= normalize_term
        # val_loss /= len(self.val_set)
        return val_loss


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        
        train_loss, val_loss, normalize_term, min_val_loss = 0, 0, 0, sys.maxsize
        loss_vec, val_loss_vec, val_vec = [], [], []
        model.train()
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.train_set):
                    feature, label = feature.to(torch.float32).to(self.device), label.to(torch.float32).to(self.device)
                    optimizer.zero_grad()
                    predict = model(feature)        
                    predict = predict.view(len(predict))
                    loss = self.criterion(predict, label)
                    normalize_term += sum(pow(label, 2)).item() / len(label)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
                
            train_loss /= normalize_term
            loss_vec.append(train_loss)
            info = f"Epoch: {epoch + 1}\tTraining NMSE: {train_loss}"
            if (epoch + 1) % self.val_step:
                self.logger.info(info)
            else:
                val_loss = self.validate()
                info += f"\tValidation NMSE: {val_loss}"
                self.logger.info(info)
                val_vec.append(epoch + 1)
                val_loss_vec.append(val_loss)

                self.save_model(self.weights_path, 'last')
                if val_loss < min_val_loss:
                    # self.lr /= 2
                    min_val_loss = val_loss
                    self.save_model(self.weights_path, 'best')
            
        self.logger.info("Training Completed.")
        self.logger.info(f"Best validation NMSE: {min_val_loss}")
        # psnr = 20 * log10(1024 / sqrt(min_val_loss))
        # self.logger.info(f"Best validation PSNR: {psnr}")

        plot_learning_curve(loss_vec, val_vec, val_loss_vec, self.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--device', type=int, nargs='+', default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--use_poison', action='store_true', help='train on patches with corrupted pixels in neighborhood')
    parser.add_argument('--data_path', type=str, default='/data1/Bad_Pixel_Correction/FiveK/feature_5')
    parser.add_argument('--model_path', type=str, default='results/mlp')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    pc.train()