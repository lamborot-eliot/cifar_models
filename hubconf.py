dependencies = ['torch']
from torchvision.models.resnet import resnet18 as _resnet18
from charles.src.effnet_models.mobilenetv2_dense import MobileNetV2 as _mobilenetv2
import torch

# resnet18 is the name of entrypoint
def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model

def mobilenetv2(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    robustMobileNetV2 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load('charles/cifar10_models/mobilenet_pgd_dense.pth'))
    return model



if __name__=="__main__":
    model = mobilenetv2(pretrained=True)
    print(model.__class__)