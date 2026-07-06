import torch
import torch.nn.functional as F
import numpy as np


EPSILON = 8 / 255.0


def clamp_img(x, x_orig, epsilon, clip_min=0.0, clip_max=1.0):
    x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
    x = torch.clamp(x, clip_min, clip_max)
    return x


def project_l2(x, x_orig, epsilon):
    diff = x - x_orig
    batch_size = diff.size(0)
    flat_diff = diff.view(batch_size, -1)
    norms = torch.norm(flat_diff, dim=1)
    
    mask = norms > epsilon
    
    if mask.any():
        scale = epsilon / norms[mask]
        for i in range(batch_size):
            if mask[i]:
                diff[i] = diff[i] * (scale[i].item() / norms[i].item())
    
    return x_orig + diff


class Attack:
    def __init__(self, model, args, proxy_loader=None):
        self.model = model
        self.args = args
        self.device = args.device
        self.epsilon = EPSILON
        self.proxy_loader = proxy_loader
    
    def fgsm(self, x, y, task_id=None):
        """
        Fast Gradient Sign Method (FGSM)
        扰动幅度: ε = 16/255
        """
        training_mode = self.model.training
        self.model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = self.model(x_adv, task_id)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + self.epsilon * grad.sign()
        self.model.train(training_mode)
        return torch.clamp(x_adv, 0, 1).detach()
    
    def pgd(self, x, y, task_id=None, steps=None):
        """
        Projected Gradient Descent (PGD)
        扰动幅度: ε = 16/255
        迭代步数: 20
        步长: 0.01
        """
        training_mode = self.model.training
        self.model.eval()
        
        _steps = steps if steps is not None else 20
        step_size = 0.01
        
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv, task_id)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        self.model.train(training_mode)
        return x_adv
    
    def mi_fgsm(self, x, y, task_id=None, steps=None):
        """
        Momentum Iterative Fast Gradient Sign Method (MI-FGSM)
        Reference: Boosting Adversarial Attacks with Momentum (CVPR 2018)
        
        方法原理：
        - 将动量算法引入迭代梯度攻击框架，通过指数衰减的累积梯度向量正则化更新方向
        - 抑制梯度的高频震荡成分，避免陷入对模型决策边界局部曲率敏感的狭窄极值区域
        - 速度向量更新规则：g_{t+1} = μ·g_t + ∇_x J(x_t, y) / ||∇_x J(x_t, y)||_1
        - 梯度采用L1归一化以消除不同迭代间梯度尺度差异
        - 沿g_{t+1}的符号方向以步长α更新对抗样本
        
        实验设置：
        - L∞扰动幅度: ε = 16/255
        - 迭代次数: T = 10
        - 动量衰减因子: μ = 1.0
        - 步长: α = ε/T = 1.6/255 ≈ 0.00627
        - 梯度归一化: L1范数
        - 损失函数: softmax交叉熵
        """
        training_mode = self.model.training
        self.model.eval()
        
        _steps = steps if steps is not None else 10
        momentum = 1.0
        step_size = self.epsilon / _steps
        
        x_adv = x.clone().detach()
        grad_accum = torch.zeros_like(x).to(self.device)
        
        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv, task_id)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            grad_l1_norm = torch.norm(grad.view(grad.size(0), -1), p=1, dim=1).view(-1, 1, 1, 1)
            grad = grad / (grad_l1_norm + 1e-8)
            
            grad_accum = momentum * grad_accum + grad
            
            with torch.no_grad():
                x_adv = x_adv + step_size * grad_accum.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        self.model.train(training_mode)
        return x_adv
    
    def df_uap(self, x, y, task_id=None):
        """
        DeepFool Universal Adversarial Perturbation (DF-UAP)
        Reference: Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations (CVPR 2020)
        
        方法原理：
        - 将DNN的logit输出向量视为输入样本在高维特征空间中的表示
        - 利用Pearson相关系数量化图像与扰动对联合特征表示的贡献度
        - 发现通用对抗扰动自身包含决定性特征，图像成分仅起噪声干扰作用
        
        损失函数：L_CL2^t = max(max_{i≠t} C_i(x+v) - C_t(x+v), -κ)
        - 仅对目标类logit施加提升压力，非目标类logit仅作为参照
        - 避免优化过程中引入非目标类特征方向的干扰梯度
        
        实验设置：
        - 使用训练数据集(2000张图片)作为代理数据集生成UAP
        - Adam优化器，学习率0.005
        - 迭代次数：200
        - L∞扰动约束：ε = 16/255
        - κ = 0 (margin)
        """
        training_mode = self.model.training
        self.model.eval()
        
        num_classes = self.args.num_classes
        lr = 0.005
        max_iter = 200
        kappa = 0.0
        
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        
        uap = torch.zeros(channel, height, width).to(self.device)
        uap.requires_grad_(True)
        
        optimizer = torch.optim.Adam([uap], lr=lr)
        
        if self.proxy_loader is not None:
            proxy_data = []
            proxy_labels = []
            for px, py in self.proxy_loader:
                proxy_data.append(px.to(self.device))
                proxy_labels.append(py.to(self.device))
            proxy_data = torch.cat(proxy_data)
            proxy_labels = torch.cat(proxy_labels)
        
        for iter_idx in range(max_iter):
            optimizer.zero_grad()
            
            if self.proxy_loader is not None:
                permuted_indices = torch.randperm(proxy_data.size(0))[:64]
                batch_x = proxy_data[permuted_indices]
                batch_y = proxy_labels[permuted_indices]
            else:
                batch_x = x
                batch_y = y
            
            x_adv = batch_x + uap.unsqueeze(0)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            outputs = self.model(x_adv, task_id)
            
            target_logits = outputs[range(batch_x.size(0)), batch_y]
            
            mask = torch.ones_like(outputs)
            mask[range(batch_x.size(0)), batch_y] = 0
            max_other_logits = torch.max(outputs * mask, dim=1)[0]
            
            loss = torch.max(max_other_logits - target_logits, torch.tensor(-kappa).to(self.device))
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                uap.data.clamp_(-self.epsilon, self.epsilon)
        
        x_adv = x + uap.detach().unsqueeze(0)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        self.model.train(training_mode)
        return x_adv
    
    def deepfool(self, x, y, task_id=None):
        """
        DeepFool Attack
        Reference: DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks (CVPR 2016)
        
        方法原理：
        - 基于分类器决策边界的局部几何性质，通过迭代线性化将非线性分类问题逐次近似为线性分类问题
        - 计算当前样本到线性化决策边界的最小欧氏距离投影
        - 对于多分类器，每次迭代对每个非当前类计算：
          w'_k = ∇f_k(x) - ∇f_{k_hat}(x)
          f'_k = f_k(x) - f_{k_hat}(x)
        - 选择距离最近的超平面：\hat{l} = argmin |f'_k|/||w'_k||₂
        - 投影向量：r_i = (|f'_{\hat{l}}|/||w'_{\hat{l}}||₂²)·w'_{\hat{l}}
        - 累加所有迭代的扰动得到总扰动r_total，乘以因子(1+η)确保越过决策边界
        
        实验设置：
        - L2范数攻击
        - 越界系数η = 0.02
        - 最大迭代次数：50（实际通常<3次收敛）
        - 鲁棒性定义：ρ_adv = (1/|D|)Σ||r_hat(x)||₂/||x||₂
        """
        training_mode = self.model.training
        self.model.eval()
        
        overshoot = 0.02
        max_iter = 50
        num_classes = self.args.num_classes
        
        x_adv = x.clone().detach()
        batch_size = x.shape[0]
        
        for b in range(batch_size):
            img = x[b:b+1].clone().detach()
            
            x_var = img.clone().detach().requires_grad_(True)
            fs = self.model(x_var, task_id)
            f_image = fs.data.cpu().numpy().flatten()
            I = f_image.argsort()[::-1][:num_classes]
            label = I[0]
            
            input_shape = img.cpu().numpy().shape
            r_tot = np.zeros(input_shape)
            
            loop_i = 0
            k_i = label
            
            while k_i == label and loop_i < max_iter:
                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)
                grad_orig = x_var.grad.data.cpu().numpy().copy()
                
                for k in range(1, num_classes):
                    if x_var.grad is not None:
                        x_var.grad.zero_()
                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = x_var.grad.data.cpu().numpy().copy()
                    
                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
                    
                    w_norm = np.linalg.norm(w_k.flatten()) + 1e-8
                    pert_k = abs(f_k) / w_norm
                    
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                
                r_i = (pert) * w / (np.linalg.norm(w) + 1e-8)
                r_tot = np.float32(r_tot + r_i)
                
                pert_image = img + torch.from_numpy(r_tot).to(self.device)
                x_var = pert_image.clone().detach().requires_grad_(True)
                fs = self.model(x_var, task_id)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())
                loop_i += 1
            
            r_tot = (1 + overshoot) * r_tot
            x_adv[b] = torch.clamp(img + torch.from_numpy(r_tot).to(self.device), 0, 1).squeeze(0)
        
        self.model.train(training_mode)
        return x_adv
    
    def cw(self, x, y, task_id=None):
        """
        Carlini-Wagner (C&W) Attack
        Reference: Towards Evaluating the Robustness of Neural Networks (Carlini & Wagner)
        
        方法原理：
        - 将对抗样本生成建模为带约束的连续优化问题
        - 基于logits的目标函数：f(x') = max(max_{i≠t} Z(x')_i - Z(x')_t, -κ)
        - κ为置信度参数，控制目标类logit超出最大非目标类logit的裕量
        - 变量替换：x' = 0.5·(tanh(w)+1)，将像素自动约束至[0,1]^n
        - 优化目标：||x'-x||₂² + c·f(x')，c通过二进制搜索动态调整
        
        实验设置：
        - L2攻击，像素值范围[0,1]
        - Adam优化器默认学习率(0.001)
        - 每子问题迭代：500步（原文为10000步，适当减少）
        - c的二进制搜索：10轮，从0.01开始
        - 置信度κ = 0
        - 初始化：w = arctanh(2x-1)使得x'=x
        """
        training_mode = self.model.training
        self.model.eval()
        
        inner_iter = 500
        learning_rate = 0.001
        kappa = 0.0
        num_classes = self.args.num_classes
        
        batch_size = x.shape[0]
        x_adv = x.clone().detach()
        
        for b in range(batch_size):
            img = x[b:b+1].clone().detach()
            original_label = y[b].item()
            
            w = torch.atanh(2 * img - 1 + 1e-8).to(self.device)
            w.requires_grad_(True)
            
            optimizer = torch.optim.Adam([w], lr=learning_rate)
            
            c_low = 0.01
            c_high = 100.0
            best_x_adv = None
            best_l2 = float('inf')
            
            for binary_round in range(10):
                c = (c_low + c_high) / 2
                
                for _ in range(inner_iter):
                    optimizer.zero_grad()
                    
                    x_try = (torch.tanh(w) + 1) / 2
                    
                    outputs = self.model(x_try, task_id)
                    
                    target_logit = outputs[0, original_label]
                    
                    mask = torch.ones(1, num_classes).to(self.device)
                    mask[0, original_label] = 0
                    max_other_logit = torch.max(outputs * mask)
                    
                    f_val = torch.max(max_other_logit - target_logit, torch.tensor(-kappa).to(self.device))
                    
                    l2_dist = torch.norm((x_try - img).view(-1)) ** 2
                    
                    loss = l2_dist + c * f_val
                    
                    loss.backward()
                    optimizer.step()
                
                x_candidate = (torch.tanh(w) + 1) / 2
                candidate_output = self.model(x_candidate, task_id)
                candidate_pred = candidate_output.argmax().item()
                
                if candidate_pred != original_label:
                    c_high = c
                    current_l2 = torch.norm((x_candidate - img).view(-1)).item()
                    if current_l2 < best_l2:
                        best_l2 = current_l2
                        best_x_adv = x_candidate.clone().detach()
                else:
                    c_low = c
            
            if best_x_adv is not None:
                x_adv[b] = best_x_adv.squeeze(0)
            else:
                x_adv[b] = ((torch.tanh(w.detach()) + 1) / 2).squeeze(0)
        
        self.model.train(training_mode)
        return x_adv
    
    def attack(self, x, y, task_id=None, attack_name=None):
        """统一攻击入口"""
        if attack_name is None:
            raise ValueError("attack_name must be specified")
        
        attack_methods = {
            'fgsm': self.fgsm,
            'pgd': self.pgd,
            'mi_fgsm': self.mi_fgsm,
            'df_uap': self.df_uap,
            'deepfool': self.deepfool,
            'cw': self.cw
        }
        
        if attack_name in attack_methods:
            return attack_methods[attack_name](x, y, task_id)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}. Available attacks: {list(attack_methods.keys())}")