import argparse
import os
import sys
import torch
from tqdm import tqdm
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from model.model import MLP_2L
from utils.logger import get_logger
from utils.plot import plot_learning_curve


class PixelCalculate():
    def __init__(self, args):
        if args.use_poison:
            train_data = SamsungDataset(args.data_path, 'poison_train')
            val_data = SamsungDataset(args.data_path, 'poison_val')
            train_path = os.path.join(args.data_path, 'poison_train')
            val_path = os.path.join(args.data_path, 'poison_val')
        else:
            train_data = SamsungDataset(args.data_path, 'train')
            val_data = SamsungDataset(args.data_path, 'val')
            train_path = os.path.join(args.data_path, 'train')
            val_path = os.path.join(args.data_path, 'val')
        
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=1, num_workers=args.num_workers, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.patch_size = len(train_data[0][0])
        self.val_step = args.val_step
        self.model = torch.nn.DataParallel(MLP_2L(self.patch_size).to(self.device))
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
        val_loss /= normalize_term
        return val_loss


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        
        train_loss, val_loss, normalize_term, min_val_loss = 0, 0, 0, sys.maxsize
        loss_vec, val_loss_vec, val_vec = [], [], []
        model.train()
        with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
            for epoch in range(self.epochs):
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
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--use_poison', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='data/medium1/feature_5')
    parser.add_argument('--model_path', type=str, default='results/mlp0')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    pc.train()