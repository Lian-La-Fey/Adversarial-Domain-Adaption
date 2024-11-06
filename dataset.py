import torch
import torchvision.transforms as T

from transformers import AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
from config import args
from PIL import Image
from glob import glob
from typing import Tuple

classes = [folder.split("\\")[-1] for folder in glob(f"{args.data_dir}/{args.source_domain}/train/*")]
classes.sort()
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

processor = AutoImageProcessor.from_pretrained(args.feature_extractor_id, use_fast=True)
transform = T.Compose([
	T.RandomHorizontalFlip(),
 	T.RandomRotation(10),
])

class OfficeDataset(Dataset):
    def __init__(self, domain, dataset, augment=False, processor=processor, transform=transform):
        self.domain = domain
        self.dataset = dataset
        self.augment = augment
        self.processor = processor
        self.transform = transform
        self.files = glob(f"{args.data_dir}/{self.domain}/{self.dataset}/*/*.jpg")
        self.labels = [class_to_idx[file.split("\\")[-2]] for file in self.files]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        label = self.labels[idx]
        if self.augment:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        return inputs, label

def get_data_loaders(domain: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = OfficeDataset(domain=domain, dataset="train", augment=True)
    val_dataset = OfficeDataset(domain=domain, dataset="val")
    test_dataset = OfficeDataset(domain=domain, dataset="test")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader