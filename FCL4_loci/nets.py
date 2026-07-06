import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 输入归一化
class NormalizationWrapper(nn.Module):
    def __init__(self, base_model):
        super(NormalizationWrapper, self).__init__()
        self.base_model = base_model
        # CIFAR-100 的均值和标准差，设为 buffer 确保其随模型移动到 GPU/CPU
        self.register_buffer('mu', torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x, task_id=None):
        # 核心步骤：在此处进行归一化
        x = (x - self.mu) / self.std
        # 转发给内部模型，task_id保留但不强制使用
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


class UnifiedFCLModel(nn.Module):
    """
    联邦持续学习模型（根据全局任务ID选择分类器分支）
    - task_idx_in_seq: 全局任务ID，用于选择分类器分支范围
    - 每个任务拥有独立的分类器分支，范围为 [task_idx_in_seq*per_task_class_num : (task_idx_in_seq+1)*per_task_class_num)
    """

    def __init__(self, args, backbone, feature_dim):
        super(UnifiedFCLModel, self).__init__()
        self.args = args
        self.features = backbone
        self.per_task_class_num = args.num_classes // args.num_tasks
        self.classifier = nn.Linear(feature_dim, args.num_classes)

    def forward(self, x, task_idx_in_seq=None):
        """
        根据全局任务ID选择分类器分支
        :param x: 输入数据
        :param task_idx_in_seq: 全局任务ID（训练的第几个任务）
        :return: 指定分支的logits输出
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        
        if task_idx_in_seq is not None:
            start = task_idx_in_seq * self.per_task_class_num
            end = (task_idx_in_seq + 1) * self.per_task_class_num
            logits = logits[:, start:end]
        
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
    else:
        args.num_classes = getattr(args, 'num_classes', 10)

    # 2. 判定骨干网络类型并加载公开权重
    model_type = getattr(args, 'model', 'cnn').lower()

    # 使用 weights='DEFAULT' 自动下载并加载官方最优的 ImageNet 预训练参数
    if 'resnet' in model_type:
        base_model = models.resnet18(weights='DEFAULT')
        feature_dim = base_model.fc.in_features
        # 截断池化层之后的输出，保留特征提取能力
        backbone = nn.Sequential(*list(base_model.children())[:-1])

    elif 'vit' in model_type:
        # 注意：ViT 权重通常针对 224x224 分辨率，若使用 CIFAR 数据集需配合 Resize
        base_model = models.vit_b_16(weights='DEFAULT')
        feature_dim = base_model.heads.head.in_features
        # 移除分类头，保留 Transformer Encoder
        backbone = nn.Sequential(*list(base_model.children())[:-1])

    elif 'mobilenet' in model_type:
        base_model = models.mobilenet_v2(weights='DEFAULT')
        feature_dim = base_model.last_channel
        backbone = base_model.features

    elif 'cnn' in model_type:
        # SimpleCNN 为本地定义，无公开预训练权重
        backbone = SimpleCNN(args)
        feature_dim = 256

    else:
        raise ValueError(f"Error: Unsupported model type '{model_type}'.")

    # 3. 使用统一分类器模型（不隔离任务头）
    model = UnifiedFCLModel(args, backbone, feature_dim)

    # 4. 使用包装器包裹模型，使其具备内置归一化功能
    model = NormalizationWrapper(model)

    print(f"Model Initialized: [{model_type.upper()}] with [ImageNet-Pretrained] weights.")
    print(f"Targeting: [{dataset_name.upper()}], Feature Dim: {feature_dim}")
    print(f"Classifier: Unified head with {args.num_classes} classes (no task isolation)")

    return model


class UnifiedKDModel(nn.Module):
    """
    Knowledge Distillation Model for EWC+KD training.
    根据全局任务ID选择分类器分支。
    """
    def __init__(self, args, feature_dim=256):
        super(UnifiedKDModel, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, feature_dim)
        
        self.per_task_class_num = args.num_classes // args.num_tasks
        self.classifier = nn.Linear(feature_dim, args.num_classes)
        
        self.register_buffer('mu', torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x, task_idx_in_seq=None):
        """
        根据全局任务ID选择分类器分支
        :param x: 输入数据
        :param task_idx_in_seq: 全局任务ID（训练的第几个任务）
        :return: 指定分支的logits输出
        """
        x = (x - self.mu) / self.std
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        logits = self.classifier(x)
        
        if task_idx_in_seq is not None:
            start = task_idx_in_seq * self.per_task_class_num
            end = (task_idx_in_seq + 1) * self.per_task_class_num
            logits = logits[:, start:end]
        
        return logits