from matplotlib import pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

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
    plt.figure()
    plt.title(f"Comparison of MLPs using different patches")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    for patch_size in nmses_dict:
        plt.plot(nmses_dict[patch_size], label=f'{patch_size}x{patch_size} MLP')
    plt.legend(loc="upper left")
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
    plt.figure()
    plt.title(f"Comparison of MLPs trained by different data")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    for model in nmses_dict:
        plt.plot(range(len(nmses_dict)),nmses_dict[model], label=f'{model}')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, 'Multimodel_NMSE.png'))