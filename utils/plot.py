from matplotlib import pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

def plot_learning_curve(loss_vec, val_vec, val_loss_vec, save_path):
    plt.figure(0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_vec)
    plt.plot(val_vec, val_loss_vec)
    plt.legend(labels=["Training", "Validation"], loc="upper right", fontsize=12)
    plt.savefig(os.path.join(save_path, 'LR_curve.png'))

def plot_NMSE(loss_vec, save_path):
    plt.figure(1)
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    plt.plot(loss_vec)
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))


def plot_mean_median(cate_vec, loss_vec, save_path):
    plt.figure(2)
    plt.xlabel('Predict Method')
    plt.ylabel('Test NMSE')
    for x, y in zip(cate_vec, loss_vec):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    plt.bar(cate_vec, loss_vec, width=0.25)
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))