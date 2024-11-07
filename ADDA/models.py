import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_extractor: nn.Module, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
    
    def forward(self, x):
        return self.feature_extractor(pixel_values=x).last_hidden_state[:, 0, :]

class Classifier(nn.Module):
    def __init__(self, inp_size: int, num_labels: int, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.classifier = nn.Linear(in_features=inp_size, out_features=num_labels)
    
    def forward(self, features):
        return self.classifier(features)

class Discriminator(nn.Module):
    def __init__(self, inp_size: int, *args, **kwargs) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.Linear(in_features=inp_size, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)