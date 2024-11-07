import torch
import itertools

from config import args, device
from tqdm import tqdm

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def train_discriminative(target_encoder, source_encoder, discriminator, source_train_loader, 
                      target_train_loader, source_val_loader, target_val_loader, 
                      optimizer_adv, optimizer_disc, criterion_adv, criterion_domain,
):
    train_adv_losses, train_disc_losses = [], []
    val_adv_losses, val_disc_losses = [], []
    num_batches = max(len(source_train_loader), len(target_train_loader))
    
    source_train_iter = itertools.cycle(iter(source_train_loader))
    target_train_iter = itertools.cycle(iter(target_train_loader))
    source_val_iter = itertools.cycle(iter(source_val_loader))
    target_val_iter = itertools.cycle(iter(target_val_loader))

    for epoch in range(args.epochs):
        discriminator.train()
        train_adv_loss = 0.0
        train_disc_loss = 0.0
        
        # -----------------------------------------------------------------------------------------
        pbar = tqdm(
            range(num_batches), 
            desc=f"Epoch {epoch+1}/{args.epochs} Training Discriminator", 
            unit="batch", total=num_batches
        )
        
        set_requires_grad(target_encoder, requires_grad=False)
        set_requires_grad(discriminator, requires_grad=True)
        for _ in pbar:
            source_inputs, _ = next(source_train_iter)
            target_inputs, _ = next(target_train_iter)
            source_inputs = source_inputs.to(device)
            target_inputs = target_inputs.to(device)
            
            optimizer_disc.zero_grad()
            
            source_encodings = source_encoder(source_inputs)
            target_encodings = target_encoder(target_inputs)
            encodings = torch.cat((source_encodings, target_encodings), dim=0)
            domain_preds = discriminator(encodings)
            
            source_domain_labels = torch.ones(source_encodings.size(0), 1).to(device)
            target_domain_labels = torch.zeros(target_encodings.size(0), 1).to(device)
            domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)
            
            disc_loss = criterion_domain(domain_preds, domain_labels)
            disc_loss.backward()
            optimizer_disc.step()
            train_disc_loss += disc_loss.item()
            pbar.set_postfix(disc_loss=f"{disc_loss.item():.4f}")
        
        # -----------------------------------------------------------------------------------------
        pbar = tqdm(
            range(len(target_train_loader)), 
            desc=f"Epoch {epoch+1}/{args.epochs} Training Encoder", 
            unit="batch", total=len(target_train_loader)
        )
        target_encoder.train()
        set_requires_grad(target_encoder, requires_grad=True)
        set_requires_grad(discriminator, requires_grad=False)
        for _ in pbar:
            target_inputs, _ = next(target_train_iter)
            target_inputs = target_inputs.to(device)
            optimizer_adv.zero_grad()
            target_features = target_encoder(target_inputs)
            domain_output = discriminator(target_features)
            source_domain_labels = torch.ones(domain_output.size(0), 1).to(device)
            adv_loss = criterion_adv(domain_output, source_domain_labels)
            adv_loss.backward()
            optimizer_adv.step()
            train_adv_loss += adv_loss.item()
            pbar.set_postfix(adv_loss=f"{adv_loss.item():.4f}")
        
        
        trian_epoch_disc_loss = train_disc_loss / num_batches
        train_epoch_adv_loss = train_adv_loss / len(target_train_loader)
        
        train_disc_losses.append(trian_epoch_disc_loss)
        train_adv_losses.append(train_epoch_adv_loss)
        
        print(f"Epoch {epoch + 1:02d} Train Discriminator Loss: {trian_epoch_disc_loss:.4f}, " 
              f"Train Adversarial Loss: {train_epoch_adv_loss:.4f}")
        
        val_adv_loss = 0.0
        val_disc_loss = 0.0
        target_encoder.eval()
        discriminator.eval()
        len_val_dataloader = max(len(source_val_loader), len(target_val_loader))
        with torch.no_grad():
            for _ in range(len_val_dataloader):
                source_inputs, _ = next(source_val_iter)
                target_inputs, _ = next(target_val_iter)
                source_inputs = source_inputs.to(device)
                target_inputs = target_inputs.to(device)
                
                source_encodings = source_encoder(source_inputs)
                target_encodings = target_encoder(target_inputs)
                encodings = torch.cat((source_encodings, target_encodings), dim=0)
                domain_preds = discriminator(encodings)
                
                source_domain_labels = torch.ones(source_encodings.size(0), 1).to(device)
                target_domain_labels = torch.zeros(target_encodings.size(0), 1).to(device)
                domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)

                loss_disc = criterion_domain(domain_preds, domain_labels)
                val_disc_loss += loss_disc.item()
            
            for _ in range(len(target_val_loader)):
                target_inputs, _ = next(target_val_iter)
                target_inputs = target_inputs.to(device)
                target_features = target_encoder(target_inputs)
                domain_output = discriminator(target_features)
                source_domain_labels = torch.ones(domain_output.size(0), 1).to(device)
                adv_loss = criterion_adv(domain_output, source_domain_labels)
                val_adv_loss += adv_loss.item()
        
        val_epoch_disc_loss = val_disc_loss / len_val_dataloader
        val_epoch_adv_loss = val_adv_loss / len_val_dataloader
        
        val_disc_losses.append(val_epoch_disc_loss)
        val_adv_losses.append(val_epoch_adv_loss)
        
        print(f"Epoch {epoch + 1:02d} Valid Discriminator Loss: {val_epoch_disc_loss:.4f}, " 
              f"Valid Adversarial Loss: {val_epoch_adv_loss:.4f}, ")

    return train_adv_losses, train_disc_losses, val_adv_losses, val_disc_losses