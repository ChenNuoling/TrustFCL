import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Tuple


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

    def set_active_task(self, task_id):
        # ❗ 级联设置任务 ID    
        if hasattr(self.base_model, 'set_active_task'):
            self.base_model.set_active_task(task_id)
    
    def get_backbone_features(self, x):
        """提取主干模型特征（用于因果模型）"""
        x = (x - self.mu) / self.std
        return self.base_model.get_backbone_features(x)
    
    def forward(self, x, task_id, use_causal=False, F_adv=None, Z=None):
        # 核心步骤：在此处进行归一化
        x = (x - self.mu) / self.std
        # 转发所有参数（包括 task_id）给内部的 FCLModel
        return self.base_model(x, task_id, use_causal, F_adv, Z)

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

class CausalModel(nn.Module):
    """
    轻量级因果模型 (2-3层MLP)
    输入: T (F_adv) 和 Z (工具变量) 的拼接
    输出: 因果特征 Y_cau (输入输出维度相同)
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 2, output_dim: int = None):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # 如果未指定输出维度，默认与输入维度相同
        self.output_dim = output_dim if output_dim is not None else feature_dim
        
        # 输入层: T和Z拼接 → 输入维度 = 2 * feature_dim
        self.input_proj = nn.Linear(2 * feature_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 输出层: 输出到指定维度
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, T: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: 对抗特征 F_adv, shape [batch, feature_dim]
            Z: 工具变量, shape [batch, feature_dim]
        Returns:
            Y_cau: 因果特征, shape [batch, output_dim]
        """
        x = torch.cat([T, Z], dim=1)
        
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
        
        Y_cau = self.output_proj(x)
        return Y_cau

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
        self.num_classes = args.num_classes
        self.num_tasks = args.num_tasks
        self.feature_dim = feature_dim  # 特征维度
        self.per_task_class_num = args.num_classes // args.num_tasks
        
        # 主干模型（完整的特征提取器）
        self.backbone = backbone
        
        # 因果模型：输入输出维度相同，都是 feature_dim
        self.causal_model = CausalModel(
            feature_dim=self.feature_dim,
            hidden_dim=getattr(args, 'causal_hidden_dim', 256),
            num_layers=args.causal_num_layers
            # 不指定 output_dim，默认与输入维度相同
        )
        
        # 统一分类器（所有任务共享）
        self.classifier = nn.Linear(self.feature_dim, args.num_classes)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取主干模型特征（二维向量）
        """
        feat = self.backbone(x)
        # 确保输出是二维向量
        if len(feat.shape) > 2:
            feat = feat.view(feat.size(0), -1)
        return feat
    
    def forward_causal_path(
        self,
        F_adv: torch.Tensor,
        F_natural: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        因果路径前向传播
        
        Args:
            F_adv: 对抗特征
            F_natural: 自然特征
        
        Returns:
            Y_cau: 因果特征
            Z: 工具变量
        """
        # 构造工具变量 Z = F_adv - F_natural（阻断梯度）
        Z = F_adv - F_natural.detach()
        
        # 因果模型提取因果特征（输入输出维度相等）
        Y_cau = self.causal_model(F_adv, Z)
        
        return Y_cau, Z

    def get_backbone_features(self, x):
        """提取主干模型特征（二维向量，用于因果模型）"""
        return self.get_features(x)
    def forward(self, x, global_task_id=None, use_causal=False, F_adv=None, Z=None):
        """
        前向传播：支持标准前向和因果前向两种模式
        
        Args:
            x: 输入数据
            global_task_id: 全局任务ID（当前训练的第几个任务，所有客户端相同）
                           根据此ID选择分类器分支范围
            use_causal: 是否使用因果模型
            F_adv: 对抗特征（用于因果推断）
            Z: 工具变量（用于因果推断）
        """
        if use_causal and F_adv is not None and Z is not None:
            # 因果推断模式：使用因果模型恢复干净特征
            Y_cau_pred = self.causal_model(F_adv, Z)
            # 通过统一分类器输出
            logits = self.classifier(Y_cau_pred)
            
            # 根据全局任务ID选择分类器分支范围
            if global_task_id is not None:
                start_idx = global_task_id * self.per_task_class_num
                end_idx = start_idx + self.per_task_class_num
                logits = logits[:, start_idx:end_idx]
            
            return logits, Y_cau_pred
        else:
            # 标准模式：直接使用主干特征，不经过因果模型
            # 这样可以避免因果模型在训练中不断累积误差
            feat = self.get_features(x)
            # 通过统一分类器输出
            logits = self.classifier(feat)
            
            # 根据全局任务ID选择分类器分支范围
            if global_task_id is not None:
                start_idx = global_task_id * self.per_task_class_num
                end_idx = start_idx + self.per_task_class_num
                logits = logits[:, start_idx:end_idx]
            
            return logits, feat

    

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
    