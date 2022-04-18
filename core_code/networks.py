import torch.nn as nn
import math

import torch


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    return sigmoid(10 * (abs(angle) / 0.7853975 - 1))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
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
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
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

        out = out + residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, aggregate_mode='last', flag_global=True, patch_size=3):
        self.inplanes = 64
        self.aggregate_mode = aggregate_mode
        self.patch_size = patch_size
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # patch差分特征比例
        self.patch_alpha = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.AdaptiveAvgPool2d(1))
        self.alpha = nn.Sequential(nn.Linear(32, 1),
                                   nn.Sigmoid())

        # patch静态特征比例
        self.patch_beta = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.AdaptiveAvgPool2d(1))
        self.beta = nn.Sequential(nn.Linear(32, 1),
                                  nn.Sigmoid())

        # 全局特征比例
        self.gamma = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())

        # 聚合特征
        if self.aggregate_mode == 'max':
            self.landa = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(512, 1),
                                       nn.Sigmoid())

        self.dropout = nn.Dropout(0.5)
        self.pred_fc1 = nn.Linear(1024, 7)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.detach().normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.detach().fill_(1)
            #     m.bias.detach().zero_()
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, points, phrase='train'):
        patch_size = self.patch_size
        if phrase == 'train':
            num_pair = 3
            point_len = points.shape[2] // 2
            # 3 [batch,128,28,28]
            basic_feature = []
            # 3 [batch,128,5,5,point_len]
            patch_feature = []
            # 2 [batch,128,5,5,point_len]
            differ_patch_feature = []

            '''----- resnet两层 获取基础特征和patch特征 -----'''
            for i in range(num_pair):
                # [batch,3,224,224]
                feature = x[:, :, :, :, i]
                feature = self.conv1(feature)
                feature = self.bn1(feature)
                feature = self.relu(feature)
                # [batch,64,112,112]
                feature = self.maxpool(feature)
                # [batch,64,56,56]
                feature = self.layer1(feature)
                # [batch,64,56,56]
                feature = self.layer2(feature)
                # [batch,128,28,28]
                basic_feature.append(feature)

                # [batch,32]
                point = points[:, i, :]
                batch_list = []
                # batch遍历
                for j in range(0, feature.shape[0]):
                    patch_list = []
                    # [128,28,28]
                    feature_batch = feature[j, :, :, :]
                    # [32]
                    point_batch = point[j, :]
                    # 点数遍历
                    for k in range(0, point_len):
                        x1 = point_batch[k * 2]
                        y1 = point_batch[k * 2 + 1]
                        patch_list.append(
                            feature_batch[:, x1 - patch_size:x1 + patch_size + 1,
                            y1 - patch_size:y1 + patch_size + 1])
                    # [128,5,5,point_len]
                    patch_list = torch.stack(patch_list, dim=3)
                    batch_list.append(patch_list)
                # [batch,128,5,5,point_len]
                batch_list = torch.stack(batch_list, dim=0)
                if i != 0:
                    differ_patch_feature.append(batch_list - patch_feature[i - 1])
                patch_feature.append(batch_list)

            '''----- 计算patch alpha-----'''
            alpha = list()
            for i in range(point_len):
                alpha.append((self.alpha(
                    torch.flatten(self.patch_alpha(differ_patch_feature[0][:, :, :, :, i]), 1, 3)) + self.alpha(
                    torch.flatten(self.patch_alpha(differ_patch_feature[1][:, :, :, :, i]), 1, 3))) / 2)
            # [batch,1,point_len]
            alpha = torch.stack(alpha, dim=2)

            '''----- 帧聚合 -----'''
            if self.aggregate_mode == 'average':
                basic_feature = (basic_feature[0] + basic_feature[1] + basic_feature[2]) / 3
                patch_feature = (patch_feature[0] + patch_feature[1] + patch_feature[2]) / 3

            if self.aggregate_mode == 'max':
                landa_list = []
                feature_list = []
                for i in range(num_pair):
                    feature = basic_feature[i]
                    feature = self.layer3(feature)
                    feature = self.layer4(feature)
                    feature = self.avgpool(feature)
                    # [batch,512]
                    feature = torch.flatten(feature, 1, 3)
                    feature_list.append(feature)
                    # 【batch,1】
                    landa = self.landa(feature)
                    landa_list.append(landa)
                landa_list = torch.stack(landa_list, dim=2)
                _, landa_list = torch.max(landa_list, dim=2)
                feature_temp = []
                batch_temp = []
                for i in range(landa_list.shape[0]):
                    index = landa_list[i][0]
                    feature_temp.append(feature_list[index][i])
                    batch_temp.append(patch_feature[index][i])
                basic_feature = torch.stack(feature_temp, dim=0)
                patch_feature = torch.stack(batch_temp, dim=0)

            if self.aggregate_mode == 'last':
                basic_feature = basic_feature[len(basic_feature) - 1]
                patch_feature = patch_feature[len(patch_feature) - 1]

            '''----- 计算patch beta-----'''
            beta = list()
            fn = list()
            for i in range(point_len):
                temp = torch.flatten(self.patch_beta(patch_feature[:, :, :, :, i]), 1, 3)
                fn.append(temp)
                beta.append(self.beta(temp))
            # [batch,1,point_len]
            beta = torch.stack(beta, dim=2)

            fn = torch.stack(fn, dim=2)
            fn = fn * (beta + alpha) / 2
            fn = torch.flatten(fn, 1, 2)

            '''----- 计算patch y-----'''
            if self.aggregate_mode != 'max':
                basic_feature = self.layer3(basic_feature)
                basic_feature = self.layer4(basic_feature)
                basic_feature = self.avgpool(basic_feature)
                basic_feature = torch.flatten(basic_feature, 1, 3)

            gamma = self.gamma(basic_feature)
            basic_feature = basic_feature * gamma
            fn = torch.cat([fn, basic_feature], dim=1)

            pred_score = self.pred_fc1(self.dropout(fn))
            return pred_score

        # if phrase == 'eval':


def resnet18_at(pretrained=False, **kwargs):
    # Constructs base a ResNet-18 model.
    model = ResNet18(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
