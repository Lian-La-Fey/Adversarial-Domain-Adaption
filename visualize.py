import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from dataset import classes
from utils import check_directory

def plot_losses(losses_dict, title):
    plt.figure(figsize=(12, 6))
    
    for label, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    check_directory("./images")
    plt.savefig(f"./images/{title}.png")
    
    plt.show()

def plot_confusion_matrix(predicts, labels, title):
    confusion = confusion_matrix(labels, predicts)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="magma", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    
    check_directory("./images")
    plt.savefig(f"./images/{title}.png")
    
    plt.show()