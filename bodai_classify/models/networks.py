import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from .stn import STN

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlockFace(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bias=False):
        super(BasicBlockFace, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU()
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNetFace(nn.Module):

    def __init__(self, block, layers, bias=False, input_shape=(3, 128, 128)):
        self.inplanes = input_shape[0]
        super(ResNetFace, self).__init__()

        # self.stn = STN(input_shape, out_dim=6)


        # self.layer1 = self._make_layer(block, 64, layers[0], stride=2, bias=bias)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bias=bias)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bias=bias)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bias=bias)
        # base_shape = get_output_shape(nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4), input_shape)
        # base_size = base_shape[0] * base_shape[1] * base_shape[2]
        # self.dropout = nn.Dropout(0.5)
        # self.fc5 = nn.Linear(base_size, 512)
        self.base = self._make_base(block, layers, bias=bias)
        base_shape = get_output_shape(self.base, input_shape)
        base_size = base_shape[0] * base_shape[1] * base_shape[2]
        self.head = self._make_head(base_size, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_base(self, block, layers, bias=False):
        layer1 = self._make_layer(block, 64, layers[0], stride=2, bias=bias)
        layer2 = self._make_layer(block, 128, layers[1], stride=2, bias=bias)
        layer3 = self._make_layer(block, 256, layers[2], stride=2, bias=bias)
        layer4 = self._make_layer(block, 512, layers[3], stride=2, bias=bias)
        return nn.Sequential(layer1, layer2, layer3, layer4)

    def _make_head(self, base_size, feature_size):
        dropout = nn.Dropout(0.5)
        fc5 = nn.Linear(base_size, feature_size)
        return nn.Sequential(dropout, fc5)

    def _make_layer(self, block, planes, blocks, stride=1, bias=False):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=3, stride=stride, padding=1, bias=bias),
                nn.BatchNorm2d(planes),
                nn.PReLU(),
            )
        layers = []
        layers.append(downsample)
        self.inplanes = planes
        for i in range(0, blocks):
            layers.append(block(self.inplanes, planes, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.stn(x)
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class LResNet(nn.Module):
    def __init__(self, block, layers, use_se=True, input_shape=(1, 128, 128)):
        self.inplanes = 64
        self.use_se = use_se
        super(LResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)

        self.base = nn.Sequential(self.conv1, self.bn1, self.prelu, self.maxpool,
                                  self.layer1, self.layer2, self.layer3, self.layer4, self.bn4)

        bn4_shape = get_output_shape(self.base, input_shape)
        bn4_size = bn4_shape[0] * bn4_shape[1] * bn4_shape[2]

        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(bn4_size, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.feature = nn.Sequential(self.dropout, self.fc5, self.bn5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        return x


def lresnet18_ir(**kwargs):
    model = LResNet(IRBlock, [2, 2, 2, 2], use_se=False, **kwargs)
    return model


def resnetface20(**kwargs):
    model = ResNetFace(BasicBlockFace, [1, 2, 4, 1], bias=False, **kwargs)
    return model


def resnetface36(**kwargs):
    model = ResNetFace(BasicBlockFace, [2, 4, 8, 2], bias=False, **kwargs)
    return model


def resnetface64(**kwargs):
    model = ResNetFace(BasicBlockFace, [3, 8, 16, 3], bias=False, **kwargs)
    return model


def get_output_shape(model, input_shape):
    batch_size = 1
    input_data = torch.rand(batch_size, *input_shape, requires_grad=False)
    output_feat = model(input_data)
    return output_feat.size()[1:]
