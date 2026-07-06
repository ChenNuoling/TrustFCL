import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


EPSILON = 8 / 255.0


def clamp_img(x, x_orig, epsilon, clip_min=0.0, clip_max=1.0):
    x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
    x = torch.clamp(x, clip_min, clip_max)
    return x


class AdvTrain:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        self.epsilon = getattr(args, 'at_epsilon', 8 / 255.0)
        self.alpha = getattr(args, 'at_alpha', 2 / 255.0)
        self.pgd_steps = getattr(args, 'at_pgd_steps', 20)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate_pgd_adv(self, x, y, steps=None):
        """生成PGD对抗样本"""
        _steps = steps if steps is not None else self.pgd_steps
        step_size = self.alpha
        
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        return x_adv
    
    def train_at(self, train_loader, optimizer, epochs):
        """
        AT: Adversarial Training (Madry et al., 2018)
        
        方法原理：
        将对抗鲁棒性问题转化为"最小-最大"博弈问题：
        - 内层最大化：在每个正常样本周围允许的扰动范围内，寻找使模型损失最大的对抗样本
        - 外层最小化：调整模型参数，使这些最坏情况下的对抗样本造成的损失最小
        
        为求解内层问题，使用投影梯度下降（PGD）：从随机起点出发，沿着损失函数的梯度方向
        （符号函数）迭代更新，每步后投影回允许的扰动球内。
        
        实验设置（CIFAR10）：
        - Wide ResNet-34-10网络
        - ε=8/255, PGD迭代7-10步, 步长2/255
        - 训练100-120轮，初始学习率0.1并在60、90轮衰减
        - 数据增强：随机水平翻转和裁剪
        - 评估：PGD-20或更强的PGD+（5个随机起点各40步）
        """
        total_time = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                x_adv = self.generate_pgd_adv(x, y)
                outputs = self.model(x_adv)
                loss = self.criterion(outputs, y)
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_trades(self, train_loader, optimizer, epochs):
        """
        TRADES: Trust Region-based Adversarial Training (Zhang et al., 2019)
        
        方法原理：
        鲁棒误差可分解为自然误差和边界误差之和。损失函数包含两项：
        - 第一项：标准交叉熵损失 CE(logits_clean, y)，确保模型在正常样本上准确分类
        - 第二项：KL散度正则项 KL(p_clean || p_adv)，约束正常样本与其对抗扰动版本
                  在模型输出分布上的差异，使模型在局部邻域内保持预测一致性
        两项通过超参数β平衡，β越大鲁棒性越强但准确率略降。
        
        实验设置（CIFAR10）：
        - ε=8/255, PGD迭代10步, 步长0.007
        - β=6（推荐默认值）
        - 训练100轮，学习率初始0.1并在75和90轮衰减
        """
        beta = getattr(self.args, 'trades_beta', 6.0)
        total_time = 0.0
        
        def generate_trades_pgd_adv(x, y):
            """TRADES专用PGD：从高斯噪声扰动起点出发"""
            step_size = 0.007
            steps = 10
            
            delta = torch.zeros_like(x).normal_() * 1e-6
            delta = delta.clamp(-self.epsilon, self.epsilon)
            x_adv = x + delta
            x_adv = torch.clamp(x_adv, 0, 1).detach()
            
            for _ in range(steps):
                x_adv.requires_grad_(True)
                outputs = self.model(x_adv)
                loss = F.cross_entropy(outputs, y)
                grad = torch.autograd.grad(loss, x_adv)[0]
                with torch.no_grad():
                    x_adv = x_adv + step_size * grad.sign()
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0, 1).detach()
            
            return x_adv
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                self.model.eval()
                x_adv = generate_trades_pgd_adv(x, y)
                self.model.train()
                
                logits_clean = self.model(x)
                logits_adv = self.model(x_adv)
                
                prob_clean = F.softmax(logits_clean, dim=1)
                prob_adv = F.softmax(logits_adv, dim=1)
                
                ce_loss = self.criterion(logits_clean, y)
                kl_loss = F.kl_div(prob_adv.log(), prob_clean, reduction='batchmean')
                
                loss = ce_loss + beta * kl_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_mart(self, train_loader, optimizer, epochs):
        """
        MART: Guided Complementary Entropy (GCE, Chen et al., 2019)
        
        方法原理：
        标准交叉熵只关注提高真实类概率，而GCE增加了对错误类概率的"中和"机制：
        
        1. 互补损失因子：计算错误类上归一化概率分布的熵（取负），最小化这项等价于
           让错误类的概率分布尽可能均匀，从而降低攻击者利用某个特定错误类进行定向
           攻击的风险。
        
        2. 引导因子：真实类概率的α次幂。当模型对当前样本预测不自信时（真实类概率小），
           互补损失的权重自动降低，避免干扰早期学习；当模型自信时则加强互补损失。
        
        损失函数：
        L_GCE = -α·log(p_y) - (1-α)·H(p_{-y})
        其中 H(p_{-y}) = -Σ_{i≠y} p_i·log(p_i) / (1-p_y) 是错误类的归一化熵
        
        这是一个"免费"的防御——替换交叉熵即可，无需额外生成对抗样本。
        也可与PGD对抗训练结合（GCE作为外层损失函数）进一步提升鲁棒性。
        
        实验设置：
        - 引导指数α统一设为1/3
        - MNIST使用LeNet-5，CIFAR10/100使用ResNet-56，Tiny ImageNet使用ResNet-50
        - SGD+momentum 0.9，权重衰减0.0001，学习率0.1在100和150轮衰减
        - 对抗评估：FGSM、BIM(10步)、PGD(40步)、MIM(40步)、JSMA、C&W攻击
        """
        alpha = getattr(self.args, 'mart_alpha', 1.0 / 3.0)
        total_time = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(x)
                prob = F.softmax(outputs, dim=1)
                
                p_y = prob[range(x.size(0)), y]
                
                mask = torch.ones_like(prob)
                mask[range(x.size(0)), y] = 0
                p_wrong = prob * mask
                p_wrong_normalized = p_wrong / (1 - p_y).view(-1, 1)
                
                log_p_wrong = torch.log(p_wrong_normalized + 1e-8)
                comp_entropy = -(p_wrong_normalized * log_p_wrong).sum(dim=1).mean()
                
                ce_loss = -torch.log(p_y + 1e-8).mean()
                
                loss = alpha * ce_loss + (1 - alpha) * comp_entropy
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_gairat(self, train_loader, optimizer, epochs):
        """
        GAIRAT: Geometric Importance-Aware Adversarial Training (Zhang et al., 2020)
        
        方法原理：
        基于两个观察：
        1. 深度模型在对抗训练中容量不足，无法拟合所有对抗样本
        2. 不同样本的重要性不同——靠近决策边界的样本（易攻击）对微调决策边界更重要，
           而远离边界的样本（难攻击）相对不重要
        
        核心设计：
        1. 几何值κ：PGD使该样本被误分类所需的最少迭代步数。κ越小表示样本越靠近边界，
           越容易被攻击，因此越重要。
        
        2. 权重函数：随κ递减的权重函数（tanh型），给κ小的样本的对抗损失赋大权重，
           给κ大的样本赋小权重。
        
        3. 预热期（burn-in）：训练初期先进行标准对抗训练（通常30-60轮），之后才启用
           重加权机制。
        
        损失函数：
        L_GAIRAT = Σ w(κ_i) · CE(logits(x+δ_i), y_i)
        其中 w(κ) = (1 - tanh(λ·κ)) / 2，λ ∈ [-1, 0]
        
        实验设置（CIFAR10）：
        - ResNet-18: ε=8/255, PGD迭代10步、步长2/255，训练100轮，学习率0.1在30和60轮衰减
          预热期30轮，λ=-1~0
        - Wide ResNet-32-10: 训练120轮，学习率在60、90、110轮衰减，预热期60轮，λ=0
        - 评估：PGD-20和PGD+（5个随机起点各40步）
        """
        burn_in_epochs = getattr(self.args, 'gairat_burn_in', 30)
        lambda_k = getattr(self.args, 'gairat_lambda', 0.0)
        total_time = 0.0
        
        def compute_geometric_value(x, y):
            """计算样本的几何值κ：PGD使样本误分类所需的最少迭代步数"""
            step_size = self.alpha
            max_steps = 20
            
            x_adv = x.clone().detach()
            k_values = torch.ones(x.size(0), device=self.device) * max_steps
            
            for step in range(max_steps):
                x_adv.requires_grad_(True)
                outputs = self.model(x_adv)
                loss = F.cross_entropy(outputs, y)
                grad = torch.autograd.grad(loss, x_adv)[0]
                
                with torch.no_grad():
                    x_adv = x_adv + step_size * grad.sign()
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0, 1).detach()
                
                preds = outputs.argmax(dim=1)
                misclassified = (preds != y) & (k_values == max_steps)
                k_values[misclassified] = step + 1
            
            return k_values
        
        def compute_weight(k_values):
            """tanh型权重函数：w(κ) = (1 - tanh(λ·κ)) / 2"""
            weights = (1 - torch.tanh(lambda_k * k_values)) / 2
            weights = weights / weights.mean()
            return weights
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            use_reweighting = (epoch >= burn_in_epochs)
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                self.model.eval()
                x_adv = self.generate_pgd_adv(x, y)
                
                if use_reweighting:
                    k_values = compute_geometric_value(x, y)
                    weights = compute_weight(k_values)
                
                self.model.train()
                logits_adv = self.model(x_adv)
                
                if use_reweighting:
                    ce_loss = F.cross_entropy(logits_adv, y, reduction='none')
                    loss = (ce_loss * weights).mean()
                else:
                    loss = self.criterion(logits_adv, y)
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_fat(self, train_loader, optimizer, epochs):
        """
        FAT: Feature-wise Adversarial Training (Cui et al., 2019)
        
        方法原理：
        标准对抗训练只在输入空间进行对抗扰动，而FAT将对抗训练扩展到特征空间。
        核心思想是：在特征空间中，对抗扰动更容易被模型学习到的特征表示所捕获。
        
        损失函数包含两项：
        - 第一项：标准交叉熵损失 CE(logits(x_adv), y)
        - 第二项：特征空间的MSE损失，约束干净样本和对抗样本的特征表示尽可能接近
        
        实验设置（CIFAR10）：
        - ResNet-18等网络架构
        - ε=8/255, PGD迭代10步
        - 训练100轮，标准配置
        """
        lambda_f = getattr(self.args, 'fat_lambda', 0.1)
        total_time = 0.0
        
        def get_features(x):
            x.requires_grad_(True)
            outputs = self.model(x)
            return outputs
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                self.model.eval()
                x_adv = self.generate_pgd_adv(x, y)
                self.model.train()
                
                feat_clean = get_features(x)
                feat_adv = get_features(x_adv)
                
                feat_diff = F.mse_loss(feat_clean, feat_adv)
                
                logits_adv = self.model(x_adv)
                ce_loss = self.criterion(logits_adv, y)
                
                loss = ce_loss + lambda_f * feat_diff
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_free_at(self, train_loader, optimizer, epochs):
        """
        Free-AT: Free Adversarial Training (Shafahi et al., 2019)
        
        方法原理：
        标准对抗训练需要额外的前向传播来生成对抗样本，而Free-AT利用训练过程中
        已经计算的梯度来更新对抗扰动，实现"免费"的对抗训练。
        
        核心思想：在每一步训练中，将当前样本视为对抗样本，计算其梯度后，沿着
        梯度方向更新扰动，得到新的对抗样本用于下一次迭代。这样无需额外的前向
        传播来生成对抗样本，大大节省计算开销。
        
        实验设置（CIFAR10）：
        - PreAct ResNet-18, Wide ResNet-34-10
        - ε=8/255, 步长2/255
        - steps_per_batch=4（每batch内更新4次扰动）
        - 训练时间约为标准AT的2/3
        """
        steps_per_batch = getattr(self.args, 'free_at_steps_per_batch', 4)
        total_time = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                x_adv = x.clone().detach()
                
                for _ in range(steps_per_batch):
                    optimizer.zero_grad()
                    
                    x_adv.requires_grad_(True)
                    outputs = self.model(x_adv)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    
                    grad = x_adv.grad.data
                    with torch.no_grad():
                        x_adv = x_adv + self.alpha * grad.sign()
                        delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                        x_adv = torch.clamp(x + delta, 0, 1).detach()
                    
                    optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_yopo(self, train_loader, optimizer, epochs):
        """
        YOPO: You Only Propagate Once (Zhang et al., 2019)
        
        方法原理：
        作者将对抗训练形式化为离散时间微分博弈，并推导其Pontryagin极大值原理（PMP）。
        PMP揭示了关键事实：对抗扰动η仅与网络第一层的参数耦合，与后续层解耦。
        
        因此更新扰动时无需完整的全网络前反向传播。核心思想：
        1. "冻结"第一层之后的梯度信息（记为p，即损失对第一层输出的梯度）
        2. 在固定p的条件下多次更新输入扰动，只需反复计算第一层的梯度
        3. YOPO-m-n：每轮先做一次完整前反向得到p（全网络传播），然后在固定p下
           对同一数据做n次第一层内的扰动更新，重复m轮后更新网络权重
        4. 访问数据m×n次，但只做m次全网络传播，相比PGD的r次全传播大幅节省计算
        
        实验设置（CIFAR10）：
        - PreAct ResNet-18 和 Wide ResNet-34-10
        - ε=8/255, 步长2/255
        - 测试YOPO-3-5和YOPO-5-3
        - 训练100轮，标准配置（学习率衰减）
        - TRADES+YOPO：YOPO-3-4比TRADES-10快2.4倍且精度更高
        """
        m = getattr(self.args, 'yopo_m', 5)    # 全传播轮数
        n = getattr(self.args, 'yopo_n', 3)    # 每轮内扰动更新次数
        total_time = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                x_adv = x.clone().detach()
                
                for _ in range(m):
                    x_adv.requires_grad_(True)
                    outputs = self.model(x_adv)
                    loss = self.criterion(outputs, y)
                    
                    grad_input = torch.autograd.grad(loss, x_adv, create_graph=True)[0]
                    
                    for _ in range(n):
                        with torch.no_grad():
                            x_adv = x_adv + self.alpha * grad_input.sign()
                            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                            x_adv = torch.clamp(x + delta, 0, 1).detach()
                        
                        x_adv.requires_grad_(True)
                        outputs = self.model(x_adv)
                        loss = self.criterion(outputs, y)
                        grad_input = torch.autograd.grad(loss, x_adv, create_graph=True)[0]
                    
                    x_adv = x_adv.detach()
                
                x_adv.requires_grad_(True)
                final_outputs = self.model(x_adv)
                final_loss = self.criterion(final_outputs, y)
                
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_shared_at(self, train_loader, optimizer, epochs):
        """
        Shared AT: Shared Adversarial Training (Mummadi et al., 2019)
        
        方法原理：
        专门针对通用扰动（Universal Perturbations）防御。理论分析显示：
        - 标准对抗训练优化的是ρ_adv（每个样本独立扰动），它是通用风险ρ_uni的上界，但不够紧
        - Shared AT提出"堆（heap）对抗训练"：将mini-batch分成大小为s的堆，对每个堆计算
          一个共享扰动同时欺骗堆内所有样本，然后将该扰动广播给堆内所有样本
        - 这相当于在ρ_adv和ρ_uni之间插值：s=1时恢复标准AT，s=d时接近通用训练
        - s越大，扰动越接近通用扰动，上界越紧
        - 为克服交叉熵的"赢者通吃"问题，引入损失截断（κ=-log 0.2）和标签平滑
        
        实验设置（CIFAR10）：
        - ResNet-20（64-128-256滤波器）
        - s∈{1,8,64}, ε从2到26变化
        - σ∈{0.3,0.5,0.7,0.9}控制准确率-鲁棒性权衡
        - 训练65轮，SGD，初始lr=0.0025（50轮后衰减10倍）
        - 4步PGD步长0.5ε
        - 结果显示s=64的Shared AT在同等准确率下鲁棒性是标准AT的2-3倍
        """
        share_size = getattr(self.args, 'shared_at_share_size', 8)  # 堆大小s
        kappa = getattr(self.args, 'shared_at_kappa', -np.log(0.2))  # 损失截断
        label_smoothing = getattr(self.args, 'shared_at_label_smoothing', 0.1)
        total_time = 0.0
        
        def truncate_loss(loss, kappa):
            """损失截断：防止交叉熵的"赢者通吃"问题"""
            return torch.clamp(loss, max=kappa)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            all_x = []
            all_y = []
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                all_x.append(x.to(self.device))
                all_y.append(y.to(self.device))
            
            if not all_x:
                continue
            
            x_all = torch.cat(all_x)
            y_all = torch.cat(all_y)
            n_samples = x_all.size(0)
            
            for i in range(0, n_samples, share_size):
                x_heap = x_all[i:i+share_size]
                y_heap = y_all[i:i+share_size]
                
                heap_size = x_heap.size(0)
                
                delta = torch.zeros(1, x_heap.size(1), x_heap.size(2), x_heap.size(3), 
                                   device=self.device, requires_grad=True)
                delta_opt = torch.optim.Adam([delta], lr=self.alpha * 10)
                
                for _ in range(4):
                    delta_opt.zero_grad()
                    x_adv = x_heap + delta
                    x_adv = torch.clamp(x_adv, 0, 1)
                    
                    outputs = self.model(x_adv)
                    smoothed_labels = torch.full_like(F.softmax(outputs, dim=1), 
                                                       label_smoothing / (self.args.num_classes - 1))
                    smoothed_labels[range(heap_size), y_heap] = 1 - label_smoothing
                    
                    loss = -torch.sum(smoothed_labels * F.log_softmax(outputs, dim=1)) / heap_size
                    loss = truncate_loss(loss, kappa)
                    
                    loss.backward()
                    delta_opt.step()
                    
                    with torch.no_grad():
                        delta.data.clamp_(-self.epsilon, self.epsilon)
                
                optimizer.zero_grad()
                x_adv_final = x_heap + delta.detach()
                x_adv_final = torch.clamp(x_adv_final, 0, 1)
                
                outputs = self.model(x_adv_final)
                smoothed_labels = torch.full_like(F.softmax(outputs, dim=1), 
                                                   label_smoothing / (self.args.num_classes - 1))
                smoothed_labels[range(heap_size), y_heap] = 1 - label_smoothing
                
                loss = -torch.sum(smoothed_labels * F.log_softmax(outputs, dim=1)) / heap_size
                loss = truncate_loss(loss, kappa)
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train_uiat(self, train_loader, optimizer, epochs):
        """
        UIAT: Inverse Adversarial Training (Dong et al., 2023)
        
        方法原理：
        标准对抗训练（如TRADES）强制对齐自然样本和对抗样本的预测分布，但当自然样本本身
        被误分类时，这种对齐会产生错误引导。为此提出逆对抗样本——通过最小化损失函数生成
        的样本，它们远离决策边界、处于高似然区域。
        
        UIAT的核心是让对抗样本去匹配其对应类别的逆对抗样本（而非原始自然样本），从而将
        对抗样本"拉向"高似然区域。具体实现：
        1. 对每个类别维护一个类别通用逆扰动z_c（单步PGD最小化该类平均损失）
        2. 用z_c给该类的所有自然样本生成逆对抗样本
        3. 记录其预测概率（并加入动量平滑）
        4. 用KL散度约束对抗样本的预测向该概率靠拢
        
        损失函数：
        L_UIAT = CE(logits(x_adv), y) + λ·KL(p_inv_momentum ∥ softmax(logits(x_adv)))
        
        实验设置（CIFAR10）：
        - ResNet-18、PreAct ResNet-18、WRN-28-10
        - ε=8/255, 逆扰动半径ε'=4/255
        - λ=3.5, β=1.0, 动量γ=0.9
        - 训练100轮，SGD+Nesterov momentum 0.9，循环学习率（最大0.1），weight decay 5e-4
        - 评估：PGD-20、CW、Auto Attack
        """
        lambda_u = getattr(self.args, 'uiat_lambda', 3.5)
        epsilon_inv = getattr(self.args, 'uiat_epsilon_inv', 4 / 255.0)
        momentum_gamma = getattr(self.args, 'uiat_momentum', 0.9)
        num_classes = self.args.num_classes
        
        class_inv_dist = torch.zeros(num_classes, num_classes, device=self.device) + 1.0 / num_classes
        
        total_time = 0.0
        
        def generate_inverse_adv(x, y):
            """生成逆对抗样本：单步PGD最小化该类平均损失"""
            x_inv = x.clone().detach().requires_grad_(True)
            outputs = self.model(x_inv)
            loss = self.criterion(outputs, y)
            grad = torch.autograd.grad(loss, x_inv)[0]
            
            with torch.no_grad():
                x_inv = x_inv - epsilon_inv * grad.sign()
                x_inv = torch.clamp(x_inv, 0, 1).detach()
            
            return x_inv
        
        for epoch in range(epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            for x, y in train_loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                self.model.eval()
                x_adv = self.generate_pgd_adv(x, y)
                x_inv = generate_inverse_adv(x, y)
                
                with torch.no_grad():
                    logits_inv = self.model(x_inv)
                    prob_inv = F.softmax(logits_inv, dim=1)
                
                for c in range(num_classes):
                    mask = (y == c)
                    if mask.any():
                        class_inv_dist[c] = momentum_gamma * class_inv_dist[c] + \
                                           (1 - momentum_gamma) * prob_inv[mask].mean(dim=0)
                
                self.model.train()
                logits_adv = self.model(x_adv)
                prob_adv = F.softmax(logits_adv, dim=1)
                
                p_inv_target = torch.gather(class_inv_dist, 0, y.view(-1, 1).expand(-1, num_classes))
                p_inv_target = p_inv_target / p_inv_target.sum(dim=1, keepdim=True)
                
                ce_loss = self.criterion(logits_adv, y)
                kl_loss = F.kl_div(prob_adv.log(), p_inv_target, reduction='batchmean')
                
                loss = ce_loss + lambda_u * kl_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
        
        return total_time
    
    def train(self, train_loader, optimizer, epochs, method='at'):
        """统一对抗训练入口"""
        method_map = {
            'at': self.train_at,
            'trades': self.train_trades,
            'mart': self.train_mart,
            'gairat': self.train_gairat,
            'fat': self.train_fat,
            'free_at': self.train_free_at,
            'yopo': self.train_yopo,
            'shared_at': self.train_shared_at,
            'uiat': self.train_uiat
        }
        
        if method not in method_map:
            raise ValueError(f"Unsupported adversarial training method: {method}. "
                           f"Available methods: {list(method_map.keys())}")
        
        return method_map[method](train_loader, optimizer, epochs)