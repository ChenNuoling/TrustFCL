import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm



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

    def forward(self, x, task_id=None):       
        x = (x - self.mu) / self.std
        
        return self.base_model(x, task_id)


class SimpleCNN(nn.Module):
    """
    基础 CNN 骨干网络，适用于轻量级实验。
    由于是自定义架构，无法直接加载 ImageNet 预训练权重。
    """

    def __init__(self, args):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
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

    def forward(self, x, global_task_id=None):
        """
        根据全局任务ID选择分类器分支范围
        global_task_id: 当前训练的第几个任务（所有客户端相同）
        例如，global_task_id为2，选择[per_task_class_num*2:per_task_class_num*3)范围的分类器分支
        映射后的标签范围都在[0, per_task_class_num)范围内
        """
        feat = self.features(x)
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
        backbone = nn.Sequential(*list(base_model.children())[:-1])
    elif 'vit' in model_type:
        base_model = models.vit_b_16(pretrained=True)
        feature_dim = base_model.heads.head.in_features
        backbone = ViTBackbone(base_model)
    elif 'mobilenet' in model_type:
        base_model = models.mobilenet_v2(pretrained=True)
        feature_dim = base_model.last_channel
        backbone = base_model.features
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
    