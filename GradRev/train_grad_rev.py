import torch
import itertools

from config import args, device
from tqdm import tqdm

def train_grad_rev(model, source_train_loader, source_val_loader, target_train_loader, 
                      target_val_loader, criterion_class, criterion_domain, optimizer, scheduler):
    train_class_losses, train_domain_losses = [], []
    val_class_losses, val_domain_losses = [], []
    num_batch = max(len(source_train_loader), len(target_train_loader))
    
    source_train_iter = itertools.cycle(iter(source_train_loader))
    target_train_iter = itertools.cycle(iter(target_train_loader))
    source_val_iter = itertools.cycle(iter(source_val_loader))
    target_val_iter = itertools.cycle(iter(target_val_loader))
    
    
    for epoch in range(args.epochs):
        model.train()
        train_class_loss = 0.0
        train_domain_loss = 0.0

        pbar = tqdm(
            range(num_batch), 
            desc=f"Epoch {epoch + 1:02d}/{args.epochs}", 
            unit="batch", total=num_batch
        )

        for i in pbar:
            
            source_inputs, source_labels = next(source_train_iter)
            target_inputs, _ = next(target_train_iter)
            source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)
            target_inputs = target_inputs.to(device)

            optimizer.zero_grad()
            
            source_class_output, source_domain_output = model(source_inputs)
            _, target_domain_output = model(target_inputs)
            domain_preds = torch.cat((source_domain_output, target_domain_output), dim=0)
            domain_labels_source = torch.ones(source_domain_output.size(0), 1).to(device)
            domain_labels_target = torch.zeros(target_domain_output.size(0), 1).to(device)
            domain_labels = torch.cat((domain_labels_source, domain_labels_target), dim=0)
            
            class_loss = criterion_class(source_class_output, source_labels)
            domain_loss = criterion_domain(domain_preds, domain_labels)
            loss = class_loss + domain_loss
            
            loss.backward()
            optimizer.step()

            train_class_loss += class_loss.item()
            train_domain_loss += domain_loss.item()

            pbar.set_postfix(batch_class_loss=f"{class_loss.item():.4f}", 
                             batch_domain_loss=f"{domain_loss.item():.4f}")
        
        train_class_epoch_loss = train_class_loss / num_batch
        train_domain_epoch_loss = train_domain_loss / num_batch
        train_class_losses.append(train_class_epoch_loss)
        train_domain_losses.append(train_domain_epoch_loss)

        print(f"Epoch {epoch + 1:02d}, "
              f"Train Class Loss: {train_class_epoch_loss:.4f}, "
              f"Train Domain Loss: {train_domain_epoch_loss:.4f}")
        
        model.eval()
        val_class_loss = 0.0
        val_domain_loss = 0.0
        len_val_dataloader = max(len(source_val_loader), len(target_val_loader))

        with torch.no_grad():
            for _ in range(len_val_dataloader):
                source_inputs, source_labels = next(source_val_iter)
                target_inputs, _ = next(target_val_iter)
                source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)
                target_inputs = target_inputs.to(device)

                source_class_output, source_domain_output = model(source_inputs)
                _, target_domain_output = model(target_inputs)
                domain_preds = torch.cat((source_domain_output, target_domain_output), dim=0)
                domain_labels_source = torch.ones(source_domain_output.size(0), 1).to(device)
                domain_labels_target = torch.zeros(target_domain_output.size(0), 1).to(device)
                domain_labels = torch.cat((domain_labels_source, domain_labels_target), dim=0)
                
                class_loss = criterion_class(source_class_output, source_labels)
                domain_loss = criterion_domain(domain_preds, domain_labels)
                
                val_class_loss += class_loss.item()
                val_domain_loss += domain_loss.item()
        
        val_class_epoch_loss = val_class_loss / len_val_dataloader
        val_domain_epoch_loss = val_domain_loss / len_val_dataloader
        val_class_losses.append(val_class_epoch_loss)
        val_domain_losses.append(val_domain_epoch_loss)
        
        scheduler.step(val_class_epoch_loss)
        
        print(f"Epoch {epoch + 1:02d}, Val Class Loss: {val_class_epoch_loss:.4f}, ",
              f"Val Domain Loss: {val_domain_epoch_loss:.4f}")
        
    return train_class_losses, train_domain_losses, val_class_losses, val_domain_losses