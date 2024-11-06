import torch

from config import args, device
from tqdm import tqdm

def train_source(model, train_loader, val_loader, criterion, optimizer, scheduler):
    train_losses, val_losses = [], []
    
    for epoch in range(args.epochs):
        model.train()
        train_running_loss = 0.0
        val_running_loss = 0.0
        
        pbar = tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch+1:02d}/{args.epochs}", 
            unit="batch", total=len(train_loader)
        )
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
        
        train_epoch_loss = train_running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)
        
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss)
        
        scheduler.step(val_epoch_loss)
        
        
        print(f"Epoch {epoch+1:02d}, Train Loss: {train_epoch_loss:.4f}, ",
              f"Valid Loss: {val_epoch_loss:.4f}")
        
    return train_losses, val_losses