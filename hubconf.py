dependencies = ['torch', 'torchvision']
from torchvision.models.resnet import resnet18 as _resnet18
from charles.src.effnet_models.mobilenetv2_dense import MobileNetV2 as _mobilenetv2
from charles.src.resnet_model.resnet_dense import ResNet34 as _resnet34

from ACL.models.resnet import resnet18 as _aclResnet18
from ACL.models.resnet_multi_bn import resnet18 as pretraining_resnet18
from ACL.models.resnet_multi_bn import proj_head

import torch
import os

import copy
import numpy as np

def cvt_state_dict(state_dict, num_classes):

    # deal with adv bn
    state_dict_new = copy.deepcopy(state_dict)

    if 1 >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(1), '')] = item


    name_to_del = []
    for name, item in state_dict_new.items():
        # print(name)
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # zero init fc
    state_dict_new['fc.weight'] = torch.zeros(
        num_classes, 512).to(state_dict['conv1.weight'].device)
    state_dict_new['fc.bias'] = torch.zeros(
        num_classes).to(state_dict['conv1.weight'].device)

    return state_dict_new

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

def resnet34_noAT(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    resnet34 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet34(**kwargs)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, 'charles/cifar10_models/resnet_dense1.pth')
        model.load_state_dict(torch.load(checkpoint))
    return model

def resnet18ACL(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    acl resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
   # Call the model, load pretrained weights
    model = pretraining_resnet18(pretrained=False, bn_names = ['normal', 'pgd'])
    ch = model.fc.in_features
    model.fc = proj_head(ch, bn_names=['normal', 'pgd'], twoLayerProj=False)
    
    
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, './ACL/downloaded_checkpoints/ACL_DS.pt')
        checkpoint = torch.load(checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    return model

def resnet18ACL_finetuned(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    acl resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _aclResnet18(num_classes=10, **kwargs)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, './ACL/downloaded_checkpoints/ACL_DS_TUNE.pt')
        checkpoint = torch.load(checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'P_state' in checkpoint:
            state_dict = checkpoint['P_state']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    return model


if __name__=="__main__":
    model = resnet18ACL(pretrained=True)
    print(model.__class__)
