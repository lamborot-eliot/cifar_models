import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, '../../super_module'))
sys.path.append(os.path.join(file_path, '../'))
sys.path.append(file_path)
import super_class
import mobilenetv2_super

class MobileNetV2(mobilenetv2_super.MobileNetV2, super_class.DeepOriginalModel):
    def __init__(self, alpha=1.0, num_classes=10):
        super(MobileNetV2, self).__init__(alpha=alpha, num_classes=num_classes)

def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

if __name__=="__main__":
    test()
