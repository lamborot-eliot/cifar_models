import torch

model = torch.hub.load("lamborot-eliot/cifar_models:main", "resnet18", pretrained=False)
print(model.__class__)
