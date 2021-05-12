dependencies = ['torch']
from torchvision.models.resnet import resnet18 as _resnet18
from charles.src.effnet_models.mobilenetv2_dense import MobileNetV2
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

def robustMobileNetV2(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    robustMobileNetV2 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = MobileNetV2()
    model.load_state_dict(torch.load('charles/cifar10_models/mobilenet_pgd_dense.pth'))
    return model



if __name__=="__main__":
    model = robustMobileNetV2()
    print(model.__class__)