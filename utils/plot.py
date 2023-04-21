from matplotlib import pyplot as plt
import os
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

def plot_learning_curve(loss_vec, val_vec, val_loss_vec, save_path):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_vec)
    plt.plot(val_vec, val_loss_vec)
    plt.legend(labels=["Training", "Validation"], loc="upper right", fontsize=12)
    plt.savefig(os.path.join(save_path, 'LR_curve.png'))

def plot_NMSE(img_size, test_loss_vec, mean_loss_vec, median_loss_vec, save_path):
    plt.figure()
    plt.title(f"Prediction using {img_size}x{img_size} patches")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    plt.plot(test_loss_vec, label='MLP')
    plt.plot(mean_loss_vec, label='Mean')
    plt.plot(median_loss_vec, label='Median')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))

def plot_multisize_NMSE(nmses_dict, save_path):
    nmses_vec = sorted(nmses_dict.items(), key=lambda x:x[0])
    nmses_dict = dict(nmses_vec)
    plt.figure()
    plt.title(f"Comparison of MLPs using different patches")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE') 
    for patch_size in nmses_dict:
        plt.plot(nmses_dict[patch_size], label=f'{patch_size}x{patch_size} patches')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig(os.path.join(save_path, 'Multisize_NMSE.png'))

def plot_mean_median(cate_vec, loss_vec, save_path):
    plt.figure()
    plt.xlabel('Predict Method')
    plt.ylabel('Test NMSE')
    for x, y in zip(cate_vec, loss_vec):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    plt.bar(cate_vec, loss_vec, width=0.25)
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))

def plot_multimodel_NMSE(nmses_dict, save_path):
    df = pd.DataFrame(nmses_dict)
    df.to_excel(os.path.join(save_path, 'nmse.xlsx'), index=False, engine='openpyxl')
    plt.figure(figsize=(8,6))
    plt.title(f"Comparison of MLPs trained by different data")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    xticks = range(len(nmses_dict['model_1']))
    plt.xticks(xticks)
    # yticks = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]
    # plt.yticks(yticks)
    plt.yscale('log')
    plt.grid()
    color_vec = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
    for i, model in enumerate(nmses_dict):
        if model == 'model_dist':
            continue
        model_order = model.split('_')[1]
        x_vec = range(len(nmses_dict[model]))
        y_vec = nmses_dict[model]
        plt.plot(x_vec, y_vec, label=f'model trained on dataset with {model_order} corrupted pixels', color=color_vec[i])
        plt.scatter(x_vec, y_vec, color=color_vec[i], marker='o', s=10)
        # for x, y in zip(x_vec, y_vec):
        #     plt.text(x, y, '%.4f' % y, ha='center', va='center', color=color_vec[i])
    
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, 'Multimodel_NMSE.png'))