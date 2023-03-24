import os
import argparse
import torch
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.plot import plot_multi_NMSE


def test(model_path, test_path, test_file):
    test_data = SamsungDataset(test_path, test_file)
    test_set = DataLoader(test_data, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path).to(device)
    model.eval()

    criterion = torch.nn.MSELoss()
    test_loss, normalize_term = 0, 0

    print("Start Testing...")
    
    with torch.no_grad():
        for i, (feature, label) in enumerate(test_set):
            feature, label = feature.to(torch.float32).to(device), label.to(torch.float32).to(device)
            predict = model(feature)
            predict = predict.view(len(predict))
            loss = criterion(predict, label)
            normalize_term += sum(pow(label, 2)).item() / len(label)
            test_loss += loss.item()  
    test_loss /= normalize_term
    print(f"Test NMSE: {test_loss}")
    print("Test Completed.")
    return test_loss


def lanuch(args):
    nmse_dict = {}
    save_path = args.model_path.split('/exp/')[0]
    for feature_dir in os.listdir(args.data_path):
        nmse_vec = []
        path_size = int(feature_dir.split('_')[1])
        
        if path_size == 5:
            model_path = args.model_path
        else:
            idx = int((path_size - args.init_size) / args.expand_size)
            model_path = os.path.join(args.model_path.split('/exp/')[0], f'exp{idx}', args.model_path.split('/exp/')[1])
        
        test_path = os.path.join(args.data_path, feature_dir)
    
        
        test_amt = len(os.listdir(os.path.join(test_path, 'corrupt'))) + 1
        for i in range(test_amt):
            if i == 0:
                standrad_test_nmse = test(model_path, test_path, 'test')
                nmse_vec.append(standrad_test_nmse)
            else:
                test_file = f'corrupt/corrupt_{i}'
                corrupt_nmse = test(model_path, test_path, test_file)
                nmse_vec.append(corrupt_nmse)
        nmse_dict[path_size] = nmse_vec
    plot_multi_NMSE(nmse_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--init_size', type=int, default=5)
    parser.add_argument('--expand_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='data/medium')
    parser.add_argument('--model_path', type=str, default='results/mlp/exp/train/weights/best.pt')
    args = parser.parse_args()
    lanuch(args)