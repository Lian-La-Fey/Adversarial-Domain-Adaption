import torch
import torch.nn as nn
import torch.autograd

class DomainClassifier(nn.Module):
    def __init__(self, inp_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)
    
class Net(nn.Module):
    def __init__(self, feature_extractor: nn.Module, inp_size: int, num_labels: int, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(in_features=inp_size, out_features=num_labels)
    
    def forward(self, x):
        features = self.feature_extractor(pixel_values=x).last_hidden_state[:, 0, :]
        class_output = self.classifier(features)
        return class_output

class AdversarialNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, inp_size: int, num_labels: int, *args, **kwargs):
        super(AdversarialNet, self).__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(in_features=inp_size, out_features=num_labels)
        self.domain_classifier = DomainClassifier(inp_size)
        self.grl = GradientReversalLayer()

    def forward(self, x):
        features = self.feature_extractor(pixel_values=x).last_hidden_state[:, 0, :]
        class_output = self.classifier(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output