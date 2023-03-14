import os
import argparse
import torch
from utils.prepocess import *
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger


def test(args):
    test_data = SamsungDataset(args.data_path, 'test')
    test_set = DataLoader(test_data, batch_size=1)

    save_path = args.model_path.split('train')[0]
    exp_dir = save_path.split('/')[2]
    test_save_path = os.path.join(save_path, 'test')
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    logger = get_logger(os.path.join(test_save_path, f'{exp_dir}_test.log'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path).to(device)
    model.eval()

    criterion = torch.nn.MSELoss()
    test_loss = 0

    logger.info("Start Testing...")
    with torch.no_grad():
        for i, (feature, label) in enumerate(test_set):
            feature, label = prepocess(feature, label)
            input, target = feature.to(device), label.to(device)
            predict = model(input)
            predict = predict.view(len(predict))
            loss = criterion(predict, target)
            test_loss += loss.item()  
    test_loss /= len(test_set)
    info = f"Test MSE: {test_loss}"
    logger.info(info)
    logger.info("Test Completed.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/medium')
    parser.add_argument('--model_path', type=str, default='results/mlp/exp/train/weights/best.pt')
    args = parser.parse_args()
    test(args)