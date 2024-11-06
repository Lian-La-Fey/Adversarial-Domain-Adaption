import torch
import torch.nn.functional as F

from config import device
from dataset import classes
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_predicted, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
    report = classification_report(all_labels, all_predicted, target_names=classes)
    print("Classification Report:")
    print(report)
    
    return all_predicted, all_labels