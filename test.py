import os
import argparse
import torch
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils.plot import plot_NMSE, plot_mean_median
import numpy as np
from tqdm import tqdm

def cal_mean_median(test_set):
    loss_mean, loss_median, normalize_term = 0, 0, 0
    for i, (feature, label) in enumerate(test_set):
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
    img_size = int(pow(len(test_data[0][0]) + 1, 0.5))
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
        with tqdm(total=len(test_set), desc=f'Test', unit='batch') as pbar:
            for i, (feature, label) in enumerate(test_set):
                feature, label = feature.to(torch.float32).to(device), label.to(torch.float32).to(device)
                predict = model(feature)
                predict = predict.view(len(predict))
                loss = criterion(predict, label)
                normalize_term += sum(pow(label, 2)).item() / len(label)
                test_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
    test_loss /= normalize_term
    logger.info(f"Test NMSE: {test_loss}")

    mean_NMSE, median_NMSE = cal_mean_median(test_set)
    logger.info(f"Mean NMSE: {mean_NMSE}")
    logger.info(f"Median NMSE: {median_NMSE}")

    if args.mode == 'test':    
        losses_vec = [mean_NMSE, median_NMSE, test_loss]
        cate_vec = ['Mean', 'Median', 'MLP']
        plot_mean_median(cate_vec, losses_vec, test_save_path)
    
    logger.info("Test Completed.")
    return img_size, test_loss, mean_NMSE, median_NMSE, test_save_path


def lanuch(args):    
    if args.mode == 'test':
        test(args, args.mode)
    elif args.mode == 'corrupt':
        NMSE_vec, mean_vec, median_vec = [], [], []
        test_amt = len(os.listdir(os.path.join(args.data_path, args.mode))) + 1
        for i in range(test_amt):
            if i == 0:
                _, standard_NMSE, standard_mean_NMSE, standard_median_NMSE, _ = test(args, 'test')
                NMSE_vec.append(standard_NMSE)
                mean_vec.append(standard_mean_NMSE)
                median_vec.append(standard_median_NMSE)
            else:
                test_file = f'{args.mode}/{args.mode}_{i}'
                img_size, corrupt_NMSE, mean_NMSE, median_NMSE, test_save_path = test(args, test_file)
                NMSE_vec.append(corrupt_NMSE)
                mean_vec.append(mean_NMSE)
                median_vec.append(median_NMSE)
        plot_NMSE(img_size, NMSE_vec, mean_vec, median_vec, test_save_path)
    else:
        print(f"No mode named {args.mode}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='corrupt')
    parser.add_argument('--data_path', type=str, default='data/medium/feature')
    parser.add_argument('--model_path', type=str, default='results/mlp/exp/train/weights/best.pt')
    args = parser.parse_args()
    lanuch(args)