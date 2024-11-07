import torch
import torch.optim as optim
import torch.nn as nn
import gc

from GradRev.train_grad_rev import train_grad_rev
from GradRev.models import Net, AdversarialNet
from dataset import get_data_loaders, class_to_idx
from config import args, device, set_seed
from train_source import train_source
from utils import check_directory, get_feature_extractor
from visualize import plot_losses, plot_confusion_matrix
from evaluate import evaluate_model

set_seed()

source_train_loader, source_val_loader, source_test_loader = get_data_loaders(
    args.source_domain, args.batch_size
)
target_train_loader, target_val_loader, target_test_loader = get_data_loaders(
    args.target_domain, args.batch_size
)

model = Net(get_feature_extractor(), inp_size=768, num_labels=len(class_to_idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)
criterion_class = nn.CrossEntropyLoss()
train_losses, val_losses = train_source(
    model, source_train_loader, source_val_loader, criterion_class, optimizer, scheduler
)

check_directory("./checkpoints")
save_path = "./checkpoints/source_model.pth"
torch.save(model.state_dict(), save_path)

plot_losses({"Training Loss": train_losses, "Validation Loss": val_losses}, title="Training Loss Over Epochs")

all_predicted, all_labels = evaluate_model(model, source_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Source Test Confusion Matrix")

all_predicted, all_labels = evaluate_model(model, target_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Target Test Confusion Matrix")

# ---------------------------- Gradient Reversal ----------------------------

torch.cuda.empty_cache()
gc.collect()

set_seed()

model = AdversarialNet(get_feature_extractor(), inp_size=768, num_labels=len(class_to_idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.BCELoss()

train_class_losses, train_domain_losses, val_class_losses, val_domain_losses = train_grad_rev(
    model, source_train_loader, source_val_loader, target_train_loader, target_val_loader,
    criterion_class, criterion_domain, optimizer, scheduler
)

check_directory("./checkpoints")
save_path = "./checkpoints/adversarial_model.pth"
torch.save(model.state_dict(), save_path)

adversarial_losses = {
    "Training Class Loss": train_class_losses,
    "Train Domain Losses": train_domain_losses,
    "Validation Class Loss": val_class_losses,
    "Validation Domain Losses": val_domain_losses,
}

plot_losses(adversarial_losses, title="Gradient Reversal - Training Loss Over Epochs")

all_predicted, all_labels = evaluate_model(model, source_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Gradient Reversal - Source Test Confusion Matrix")

all_predicted, all_labels = evaluate_model(model, target_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Gradient Reversal - Target Test Confusion Matrix")