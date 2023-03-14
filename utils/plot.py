from matplotlib import pyplot as plt
import os

def plot_learning_curve(loss_vec, val_vec, val_loss_vec, save_path):
    plt.figure(0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_vec)
    plt.plot(val_vec, val_loss_vec)
    plt.legend(labels=["Training", "Validation"], loc="upper right", fontsize=12)
    plt.savefig(os.path.join(save_path, 'LR_curve.png'))