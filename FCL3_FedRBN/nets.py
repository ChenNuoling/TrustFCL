import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

'''
6/7更新
为CNN、ResNet、mobilenet模型添加双分支bn层
'''

# 输入归一化
class NormalizationWrapper(nn.Module):
    def __init__(self, base_model, dataset='cifar100', model_type='resnet'):
        super(NormalizationWrapper, self).__init__()
        self.base_model = base_model
        self.model_type = model_type.lower()
        
        dataset = dataset.lower()
        
        if 'vit' in self.model_type:
            mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        elif 'cifar100' in dataset:
            mu = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
        elif 'cifar10' in dataset:
            mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        elif 'mnist' in dataset:
            mu = torch.tensor([0.1307]).view(1, 1, 1, 1)
            std = torch.tensor([0.3081]).view(1, 1, 1, 1)
        elif 'imagenet' in dataset:
            mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            mu = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
        
        self.register_buffer('mu', mu)
        self.register_buffer('std', std)

    def forward(self, x, task_id=None, use_bn_a=False):       
        x = (x - self.mu) / self.std
        
        return self.base_model(x, task_id, use_bn_a=use_bn_a)


class SimpleCNN(nn.Module):
    """
    基础 CNN 骨干网络，支持FedRBN双分支批归一化（Dual BN）。
    BN_c: 用于干净样本, BN_a: 用于对抗样本
    """

    def __init__(self, args):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_c = nn.BatchNorm2d(32)
        self.bn1_a = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_c = nn.BatchNorm2d(64)
        self.bn2_a = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_c = nn.BatchNorm2d(128)
        self.bn3_a = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

    def forward(self, x, use_bn_a=False):
        if use_bn_a:
            x = self.pool(F.relu(self.bn1_a(self.conv1(x))))
            x = self.pool(F.relu(self.bn2_a(self.conv2(x))))
            x = self.pool(F.relu(self.bn3_a(self.conv3(x))))
        else:
            x = self.pool(F.relu(self.bn1_c(self.conv1(x))))
            x = self.pool(F.relu(self.bn2_c(self.conv2(x))))
            x = self.pool(F.relu(self.bn3_c(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x

    def get_bn_c_mean_var(self):
        return {
            'bn1': (self.bn1_c.running_mean, self.bn1_c.running_var),
            'bn2': (self.bn2_c.running_mean, self.bn2_c.running_var),
            'bn3': (self.bn3_c.running_mean, self.bn3_c.running_var),
        }


class ResNetDualBN(nn.Module):
    """
    支持FedRBN双分支批归一化的ResNet骨干网络。
    将原始ResNet中的每个BN层替换为双分支BN（BN_c用于干净样本，BN_a用于对抗样本）
    """

    def __init__(self, base_model):
        super(ResNetDualBN, self).__init__()
        self.conv1 = base_model.conv1
        self.bn1_c = base_model.bn1
        self.bn1_a = nn.BatchNorm2d(self.bn1_c.num_features)
        self.bn1_a.load_state_dict(self.bn1_c.state_dict())
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = self._convert_layer(base_model.layer1)
        self.layer2 = self._convert_layer(base_model.layer2)
        self.layer3 = self._convert_layer(base_model.layer3)
        self.layer4 = self._convert_layer(base_model.layer4)

    def _convert_layer(self, layer):
        new_blocks = []
        for block in layer:
            new_block = self._convert_block(block)
            new_blocks.append(new_block)
        return nn.Sequential(*new_blocks)

    def _convert_block(self, block):
        new_block = nn.Module()
        new_block.conv1 = block.conv1
        new_block.bn1_c = block.bn1
        new_block.bn1_a = nn.BatchNorm2d(block.bn1.num_features)
        new_block.bn1_a.load_state_dict(block.bn1.state_dict())
        new_block.relu = block.relu
        new_block.conv2 = block.conv2
        new_block.bn2_c = block.bn2
        new_block.bn2_a = nn.BatchNorm2d(block.bn2.num_features)
        new_block.bn2_a.load_state_dict(block.bn2.state_dict())
        
        if hasattr(block, 'downsample') and block.downsample is not None:
            new_block.downsample = nn.Module()
            new_block.downsample.conv = block.downsample[0]
            new_block.downsample.bn_c = block.downsample[1]
            new_block.downsample.bn_a = nn.BatchNorm2d(block.downsample[1].num_features)
            new_block.downsample.bn_a.load_state_dict(block.downsample[1].state_dict())
        else:
            new_block.downsample = None
        
        new_block.stride = block.stride
        return new_block

    def forward(self, x, use_bn_a=False):
        x = self.conv1(x)
        if use_bn_a:
            x = self.bn1_a(x)
        else:
            x = self.bn1_c(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self._forward_layer(self.layer1, x, use_bn_a)
        x = self._forward_layer(self.layer2, x, use_bn_a)
        x = self._forward_layer(self.layer3, x, use_bn_a)
        x = self._forward_layer(self.layer4, x, use_bn_a)

        return x

    def _forward_layer(self, layer, x, use_bn_a):
        for block in layer:
            identity = x

            out = block.conv1(x)
            if use_bn_a:
                out = block.bn1_a(out)
            else:
                out = block.bn1_c(out)
            out = block.relu(out)

            out = block.conv2(out)
            if use_bn_a:
                out = block.bn2_a(out)
            else:
                out = block.bn2_c(out)

            if block.downsample is not None:
                identity = block.downsample.conv(identity)
                if use_bn_a:
                    identity = block.downsample.bn_a(identity)
                else:
                    identity = block.downsample.bn_c(identity)

            out += identity
            out = block.relu(out)

            x = out
        return x


class ViTBackbone(nn.Module):
    """
    ViT骨干网络封装，正确处理输入并提取特征
    """
    def __init__(self, base_model):
        super(ViTBackbone, self).__init__()
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        self.encoder = base_model.encoder
        
    def forward(self, x):
        x = self.conv_proj(x)
        n = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        
        return x


class MobileNetDualBN(nn.Module):
    """
    支持FedRBN双分支批归一化的MobileNetV2骨干网络。
    将原始MobileNetV2中的每个BN层替换为双分支BN（BN_c用于干净样本，BN_a用于对抗样本）
    """

    def __init__(self, base_model):
        super(MobileNetDualBN, self).__init__()
        features = base_model.features
        
        self.conv1 = features[0][0]
        self.bn1_c = features[0][1]
        self.bn1_a = nn.BatchNorm2d(self.bn1_c.num_features)
        self.bn1_a.load_state_dict(self.bn1_c.state_dict())
        self.relu = features[0][2]
        
        self.layers = nn.ModuleList()
        for i in range(1, len(features)):
            block = features[i]
            if hasattr(block, 'conv'):
                new_block = self._convert_inverted_residual(block)
            else:
                new_block = self._convert_conv_block(block)
            self.layers.append(new_block)

    def _convert_conv_block(self, block):
        new_block = nn.Module()
        new_block.conv = block[0]
        new_block.bn_c = block[1]
        new_block.bn_a = nn.BatchNorm2d(block[1].num_features)
        new_block.bn_a.load_state_dict(block[1].state_dict())
        new_block.relu = block[2] if len(block) > 2 else None
        return new_block

    def _convert_inverted_residual(self, block):
        new_block = nn.Module()
        new_block.use_res_connect = block.use_res_connect
        
        new_block.conv = nn.ModuleList()
        for conv_module in block.conv:
            sub_block = nn.Module()
            if hasattr(conv_module, 'conv'):
                sub_block.conv = conv_module.conv
                sub_block.bn_c = conv_module.bn
                sub_block.bn_a = nn.BatchNorm2d(conv_module.bn.num_features)
                sub_block.bn_a.load_state_dict(conv_module.bn.state_dict())
                sub_block.relu = conv_module.relu
            else:
                sub_block.conv = conv_module[0]
                sub_block.bn_c = conv_module[1]
                sub_block.bn_a = nn.BatchNorm2d(conv_module[1].num_features)
                sub_block.bn_a.load_state_dict(conv_module[1].state_dict())
                sub_block.relu = conv_module[2] if len(conv_module) > 2 else None
            new_block.conv.append(sub_block)
        
        return new_block

    def forward(self, x, use_bn_a=False):
        x = self.conv1(x)
        if use_bn_a:
            x = self.bn1_a(x)
        else:
            x = self.bn1_c(x)
        if self.relu:
            x = self.relu(x)
        
        for layer in self.layers:
            if hasattr(layer, 'conv') and hasattr(layer, 'use_res_connect'):
                x = self._forward_inverted_residual(layer, x, use_bn_a)
            else:
                x = self._forward_conv_block(layer, x, use_bn_a)
        
        return x

    def _forward_conv_block(self, block, x, use_bn_a):
        x = block.conv(x)
        if use_bn_a:
            x = block.bn_a(x)
        else:
            x = block.bn_c(x)
        if block.relu:
            x = block.relu(x)
        return x

    def _forward_inverted_residual(self, block, x, use_bn_a):
        identity = x
        
        for sub_block in block.conv:
            x = sub_block.conv(x)
            if use_bn_a:
                x = sub_block.bn_a(x)
            else:
                x = sub_block.bn_c(x)
            if sub_block.relu:
                x = sub_block.relu(x)
        
        if block.use_res_connect:
            x += identity
        
        return x


class UnifiedFCLModel(nn.Module):
    """
    统一分类器的联邦持续学习模型（根据全局任务ID选择分类器分支）
    所有任务共享同一个分类器，但根据global_task_id（全局任务索引）选择特定范围的输出
    全局任务ID：当前训练的第几个任务，所有客户端相同
    每个客户端的任务序列是随机的，但同一轮训练时所有客户端使用相同的分类器分支
    """

    def __init__(self, args, backbone, feature_dim):
        super(UnifiedFCLModel, self).__init__()
        self.args = args
        self.features = backbone
        self.per_task_class_num = args.num_classes // args.num_tasks
        self.classifier = nn.Linear(feature_dim, args.num_classes)

    def forward(self, x, global_task_id=None, use_bn_a=False):
        """
        根据全局任务ID选择分类器分支范围
        global_task_id: 当前训练的第几个任务（所有客户端相同）
        例如，global_task_id为2，选择[per_task_class_num*2:per_task_class_num*3)范围的分类器分支
        映射后的标签范围都在[0, per_task_class_num)范围内
        use_bn_a: 是否使用对抗样本分支的BN层
        """
        feat = self.features(x, use_bn_a=use_bn_a)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        
        if global_task_id is not None:
            start_idx = global_task_id * self.per_task_class_num
            end_idx = start_idx + self.per_task_class_num
            logits = logits[:, start_idx:end_idx]
        
        return logits


def Init_model(args):
    """
    模型工厂函数：根据配置参数初始化并返回模型实例。
    """
    # 1. 判定类别数
    dataset_name = getattr(args, 'dataset', '').lower()
    if 'cifar100' in dataset_name:
        args.num_classes = 100
    elif 'cifar10' in dataset_name:
        args.num_classes = 10
    elif 'mnist' in dataset_name:
        args.num_classes = 10
    else:
        args.num_classes = getattr(args, 'num_classes', 10)

    # 2. 判定骨干网络类型并加载公开权重
    model_type = getattr(args, 'model', 'cnn').lower()

    if 'resnet' in model_type:
        base_model = models.resnet18(pretrained=True)
        feature_dim = base_model.fc.in_features
        backbone = ResNetDualBN(base_model)
    elif 'vit' in model_type:
        base_model = models.vit_b_16(pretrained=True)
        feature_dim = base_model.heads.head.in_features
        backbone = ViTBackbone(base_model)
    elif 'mobilenet' in model_type:
        base_model = models.mobilenet_v2(pretrained=True)
        feature_dim = base_model.last_channel
        backbone = MobileNetDualBN(base_model)
    elif 'cnn' in model_type:
        backbone = SimpleCNN(args)
        feature_dim = 256
    else:
        raise ValueError(f"Error: Unsupported model type '{model_type}'.")

    # 3. 使用统一分类器模型（不隔离任务头）
    model = UnifiedFCLModel(args, backbone, feature_dim)

    # 4. 使用包装器包裹模型，使其具备内置归一化功能
    model = NormalizationWrapper(model, dataset=dataset_name, model_type=model_type)

    print(f"Model Initialized: [{model_type.upper()}] with [ImageNet-Pretrained] weights.")
    print(f"Targeting: [{dataset_name.upper()}], Feature Dim: {feature_dim}")
    print(f"Classifier: Unified head with {args.num_classes} classes (no task isolation)")
    
    return model
    