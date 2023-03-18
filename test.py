import os
import argparse
import torch
from utils.prepocess import *
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils.plot import plot_NMSE

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

            input, target = feature.to(device), label.to(device)
            predict = model(input)
            predict = predict.view(len(predict))
            loss = criterion(predict, target)
            normalize_term += sum(pow(target, 2)).item() / len(target)
            test_loss += loss.item()  
    test_loss /= normalize_term
    info = f"Test NMSE: {test_loss}"
    logger.info(info)
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
    parser.add_argument('--mode', type=str, default='corrupt')
    parser.add_argument('--data_path', type=str, default='data/medium')
    parser.add_argument('--model_path', type=str, default='results/mlp/exp/train/weights/best.pt')
    args = parser.parse_args()
    lanuch(args)