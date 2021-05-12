import torch

model = torch.hub.load("lamborot-eliot/cifar_models:main", "mobilenetv2", pretrained=True)
print(model.__class__)
model = torch.hub.load("lamborot-eliot/cifar_models:main", "resnet34", pretrained=True)
print(model.__class__)