dependencies = ['torch', 'torchvision']
from torchvision.models.resnet import resnet18 as _resnet18
from charles.src.effnet_models.mobilenetv2_dense import MobileNetV2 as _mobilenetv2
from charles.src.resnet_model.resnet_dense import ResNet34 as _resnet34

import torch
import os

def mobilenetv2(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    mobilenetv2 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, 'charles/cifar10_models/mobilenet_pgd_dense.pth')
        model.load_state_dict(torch.load(checkpoint))
    return model

def resnet34(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    resnet34 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet34(**kwargs)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, 'charles/cifar10_models/resnet_pgd.pth')
        model.load_state_dict(torch.load(checkpoint))
    return model


if __name__=="__main__":
    model = resnet34(pretrained=True)
    print(model.__class__)