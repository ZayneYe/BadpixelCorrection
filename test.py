import os
import argparse
import torch
from utils.prepocess import *
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils.plot import plot_NMSE, plot_mean_median
import numpy as np

def cal_mean_median(test_set):
    loss_mean, loss_median, normalize_term = 0, 0, 0
    for i, feature in enumerate(test_set):
        feature, label = prepocess(feature)
        input, target = feature.numpy(), label.numpy()

        input_mean = np.mean(input, axis=1)
        loss_mean += np.mean(pow(input_mean - target, 2))
        
        input_median = np.median(input, axis=1)
        loss_median += np.mean(pow(input_median - target, 2))
        
        normalize_term += np.mean(pow(target, 2))
    
    return loss_mean / normalize_term, loss_median / normalize_term
        

def test(args, test_file):
    test_data = SamsungDataset(args.data_path, test_file)
    test_set = DataLoader(test_data, batch_size=1)
    
    save_path = args.model_path.split('train')[0]
    exp_dir = save_path.split('/')[2]
    test_save_path = os.path.join(save_path, args.mode)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    logger = get_logger(os.path.join(test_save_path, f'{exp_dir}_{args.mode}.log'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path).to(device)
    model.eval()

    criterion = torch.nn.MSELoss()
    test_loss, normalize_term = 0, 0

    logger.info("Start Testing...")
    with torch.no_grad():
        for i, feature in enumerate(test_set):
            feature, label = prepocess(feature)
            feature, label = feature.to(device), label.to(device)
            predict = model(feature)
            predict = predict.view(len(predict))
            loss = criterion(predict, label)
            normalize_term += sum(pow(label, 2)).item() / len(label)
            test_loss += loss.item()  
    test_loss /= normalize_term
    logger.info(f"Test NMSE: {test_loss}")
    if test_file == 'test':
        mean_NMSE, median_NMSE = cal_mean_median(test_set)
        logger.info(f"Mean NMSE: {mean_NMSE}")
        logger.info(f"Median NMSE: {median_NMSE}")
        losses_vec = [mean_NMSE, median_NMSE, test_loss]
        cate_vec = ['Mean', 'Median', 'MLP']
        plot_mean_median(cate_vec, losses_vec, test_save_path)

    logger.info("Test Completed.")
    return test_loss, test_save_path


def lanuch(args):
    if args.mode == 'test':
        test(args, args.mode)
    elif args.mode == 'corrupt':
        NMSE_vec = []
        test_amt = len(os.listdir(os.path.join(args.data_path, 'feature', args.mode))) + 1
        for i in range(test_amt):
            if i == 0:
                standard_NMSE, _ = test(args, 'test')
                NMSE_vec.append(standard_NMSE)
            else:
                test_file = f'{args.mode}/{args.mode}_{i}'
                corrupt_NMSE, test_save_path = test(args, test_file)
                NMSE_vec.append(corrupt_NMSE)
        plot_NMSE(NMSE_vec, test_save_path)
    else:
        print(f"No mode named {args.mode}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--data_path', type=str, default='data/medium')
    parser.add_argument('--model_path', type=str, default='results/mlp/exp/train/weights/best.pt')
    args = parser.parse_args()
    lanuch(args)