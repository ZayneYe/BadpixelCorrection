import argparse
import os
import sys
import torch
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from model.model import MLP_2L
from utils.logger import get_logger
from utils.prepocess import *
from utils.plot import plot_learning_curve


class PixelCalculate():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.image_size = args.image_size
        self.val_step = args.val_step
        self.model = MLP_2L(self.image_size).to(self.device)
        self.model_path = args.model_path
    
        train_data = SamsungDataset(args.data_path, 'train')
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        val_data = SamsungDataset(args.data_path, 'val')
        self.val_set = DataLoader(val_data, batch_size=1, shuffle=False)

        self.criterion = torch.nn.MSELoss()

        idx = 1
        exp_dir = f'exp{idx}'
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
            for i, feature in enumerate(self.val_set):
                feature, label = prepocess(feature)
                input, target = feature.to(self.device), label.to(self.device)
                predict = self.model(input)
                predict = predict.view(len(predict))
                loss = self.criterion(predict, target)
                normalize_term += sum(pow(target, 2)).item() / len(target)
                val_loss += loss.item()  
        val_loss /= normalize_term
        return val_loss


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        
        train_loss, val_loss, normalize_term, min_val_loss = 0, 0, 0, sys.maxsize
        loss_vec, val_loss_vec, val_vec = [], [], []
        model.train()
        for epoch in range(self.epochs):
            for i, feature in enumerate(self.train_set):
                feature, label = prepocess(feature)
                feature, label = feature.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                predict = model(feature)
                predict = predict.view(len(predict)) 
                loss = self.criterion(predict, label)
                normalize_term += sum(pow(label, 2)).item() / len(label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            
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
        plot_learning_curve(loss_vec, val_vec, val_loss_vec, self.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=5)
    parser.add_argument('--val_step', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data/medium')
    parser.add_argument('--model_path', type=str, default='results/mlp')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    pc.train()
