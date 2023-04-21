import os
import argparse
import torch
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from utils.plot import plot_multisize_NMSE, plot_multimodel_NMSE


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


def multisize_lanuch(args):
    nmse_dict = {}
    save_path = os.path.join(args.model_path.split('/')[0], args.mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
    plot_multisize_NMSE(nmse_dict, save_path)


def multimodel_lanuch(args):
    model_amt, corrupt_amt = 6, 5
    save_path = os.path.join(args.model_path.split('/')[0], args.mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    nmse_dict = {}
    for i in range(model_amt):
        if i == 5:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.model_path.split('/mlp_dist/')[0], f'mlp{i}', args.model_path.split('/mlp_dist/')[1])
        nmse_vec = []
        for j in range(corrupt_amt):
            test_path = os.path.join(args.data_path, f'feature_5')
            if j == 0:
                standrad_test_nmse = test(model_path, test_path, 'test')
                nmse_vec.append(standrad_test_nmse)
            else:
                test_file = f'corrupt/corrupt_{j}'
                corrupt_nmse = test(model_path, test_path, test_file)
                nmse_vec.append(corrupt_nmse)
        if i == 5:
            nmse_dict[f'model_dist'] = nmse_vec
        else:     
            nmse_dict[f'model_{i}'] = nmse_vec
    plot_multimodel_NMSE(nmse_dict, save_path)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='multimodel')
    parser.add_argument('--init_size', type=int, default=5)
    parser.add_argument('--expand_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='data/medium1')  
    parser.add_argument('--model_path', type=str, default='results2/mlp_dist/exp/train/weights/best.pt')
    args = parser.parse_args()
    if args.mode == 'multisize':
        multisize_lanuch(args)
    elif args.mode == 'multimodel':
        multimodel_lanuch(args)