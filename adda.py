import torch
import torch.optim as optim
import torch.nn as nn

from ADDA.train_adda import set_requires_grad, train_discriminative
from dataset import get_data_loaders, class_to_idx
from utils import check_directory, get_feature_extractor
from ADDA.models import Encoder, Classifier, Discriminator
from config import args, device, set_seed
from train_source import train_source
from visualize import plot_losses, plot_confusion_matrix
from evaluate import evaluate_model

set_seed()

source_train_loader, source_val_loader, source_test_loader = get_data_loaders(
    args.source_domain, args.batch_size
)
target_train_loader, target_val_loader, target_test_loader = get_data_loaders(
    args.target_domain, args.batch_size
)

source_encoder = Encoder(get_feature_extractor()).to(device)
classifier = Classifier(inp_size=768, num_labels=len(class_to_idx)).to(device)
model = nn.Sequential(source_encoder, classifier)
optimizer_cls = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion_cls = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_cls, mode='min', factor=0.5, patience=2, verbose=True
)

train_losses, val_losses = train_source(
    model, source_train_loader, source_val_loader, criterion_cls, optimizer_cls, scheduler
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

set_seed()

set_requires_grad(source_encoder, requires_grad=False)
set_requires_grad(classifier, requires_grad=False)

target_encoder = Encoder(get_feature_extractor()).to(device)
target_encoder.load_state_dict(source_encoder.state_dict())
discriminator = Discriminator(inp_size=768).to(device)

optimizer_disc = optim.Adam(discriminator.parameters(), lr=args.adv_learning_rate)
optimizer_adv = optim.Adam(target_encoder.parameters(), lr=args.adv_learning_rate)

criterion_domain = nn.BCELoss()
criterion_adv = nn.BCELoss()

train_adv_losses, train_disc_losses, val_adv_losses, val_disc_losses = train_discriminative(
    target_encoder, source_encoder, 
    discriminator, source_train_loader, 
    target_train_loader, source_val_loader, target_val_loader, 
    optimizer_adv, optimizer_disc, criterion_adv, criterion_domain,
)

model = nn.Sequential(target_encoder, classifier)
save_path = "./checkpoints/discriminative_model.pth"
torch.save(model.state_dict(), save_path)

adversarial_losses = {
    "Train Adversarial Loss": train_adv_losses,
    "Train Discriminator Losses": train_disc_losses,
    "Validation Adversarial Loss": val_adv_losses,
    "Validation Discriminator Losses": val_disc_losses,
}

plot_losses(adversarial_losses, title="Discriminative - Training Loss Over Epochs")

all_predicted, all_labels = evaluate_model(model, source_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Discriminative - Source Test Confusion Matrix")

all_predicted, all_labels = evaluate_model(model, target_test_loader)
plot_confusion_matrix(all_predicted, all_labels, title="Discriminative - Target Test Confusion Matrix")