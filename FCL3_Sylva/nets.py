import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class LoRALayer(nn.Module):
    """LoRA 层：冻结原参数，只训练低秩分解矩阵 A 和 B"""
    def __init__(self, in_features, out_features, rank=4, alpha=16, layer_type='linear', stride=1, padding=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        
        if layer_type == 'linear':
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
        elif layer_type == 'conv2d':
            self.lora_A = nn.Conv2d(in_features, rank, kernel_size=1, stride=stride, padding=padding, bias=False)
            self.lora_B = nn.Conv2d(rank, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.lora_B.weight)
        
        self.use_lora = True
    
    def forward(self, x):
        if self.use_lora:
            return self.lora_B(self.lora_A(x)) * self.alpha / self.rank
        return 0


class LoRAWrapper(nn.Module):
    """将 LoRA 包装器包装在原层外"""
    def __init__(self, base_layer, lora_rank=4, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        
        # 暴露 weight 和 bias 属性，避免 MultiheadAttention 内部访问报错
        self.weight = base_layer.weight
        self.bias = base_layer.bias if hasattr(base_layer, 'bias') else None
        
        # 冻结原始层参数
        for param in base_layer.parameters():
            param.requires_grad = False
        
        if isinstance(base_layer, nn.Linear):
            self.lora = LoRALayer(base_layer.in_features, base_layer.out_features, 
                                 rank=lora_rank, alpha=lora_alpha, layer_type='linear')
        elif isinstance(base_layer, nn.Conv2d):
            self.lora = LoRALayer(base_layer.in_channels, base_layer.out_channels, 
                                 rank=lora_rank, alpha=lora_alpha, layer_type='conv2d',
                                 stride=base_layer.stride, padding=base_layer.padding)
    
    def forward(self, x):
        return self.base_layer(x) + self.lora(x)


def add_lora_to_vit_qkv(model, lora_rank=4):
    """
    为 ViT 的关键层添加 LoRA
    torchvision ViT 使用 MultiheadAttention，这里为 MLP 层和分类器添加 LoRA
    """
    count = 0
    
    # 递归遍历所有模块
    for name, module in model.named_modules():
        # 处理 MLP 层 (linear_1 和 linear_2)
        if 'mlp' in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            try:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRAWrapper(module, lora_rank=lora_rank))
                count += 1
                print(f"✓ 添加 LoRA: {name}")
            except Exception as e:
                print(f"✗ 失败 {name}: {e}")
        # 处理分类器
        elif name.endswith('classifier'):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                try:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, LoRAWrapper(module, lora_rank=lora_rank))
                    count += 1
                    print(f"✓ 添加 LoRA: {name}")
                except Exception as e:
                    print(f"✗ 失败 {name}: {e}")
    
    print(f"总共添加 {count} 个 LoRA 层")
    return model


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


class FCLModel(nn.Module):
    """
    统一分类器的联邦持续学习模型（根据全局任务ID选择分类器分支）
    """

    def __init__(self, args, backbone, feature_dim):
        super(FCLModel, self).__init__()
        self.args = args
        self.features = backbone
        self.per_task_class_num = args.num_classes // args.num_tasks
        self.classifier = nn.Linear(feature_dim, args.num_classes)

    def forward(self, x, global_task_id=None):
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

    # 3. 封装为支持 FCL 的多头模型
    model = FCLModel(args, backbone, feature_dim)

    # 4. 使用包装器包裹模型，使其具备内置归一化功能
    model = NormalizationWrapper(model, dataset=dataset_name, model_type=model_type)
    
    # 5. 添加 LoRA 参数
    if 'vit' in model_type:
        model = add_lora_to_vit_qkv(model, lora_rank=args.lora_rank)
    else:
        # 对于非 ViT 模型，使用原来的方法
        def add_lora_to_model(module, lora_rank=4):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRAWrapper(child, lora_rank=lora_rank))
                else:
                    add_lora_to_model(child, lora_rank)
            return module
        model = add_lora_to_model(model, lora_rank=args.lora_rank)

    print(f"Model Initialized: [{model_type.upper()}] with [ImageNet-Pretrained] weights.")
    print(f"Targeting: [{dataset_name.upper()}], Feature Dim: {feature_dim}")

    return model