import torch

model = torch.hub.load("lamborot-eliot/cifar_models:main", "resnet18", pretrained=False)
print(model.__class__)
model = torch.hub.load("lamborot-eliot/cifar_models:main", "mobilenetv2", pretrained=False)
print(model.__class__)
