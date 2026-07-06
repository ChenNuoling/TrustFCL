import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from utils import get_memory_usage



class Client:
    def __init__(self, cid, args, model, data_loader, task_sequence):
        self.cid = cid
        self.args = args
        self.model = model.to(args.device)
        self.data_generator = data_loader
        self.task_sequence = task_sequence
        self.class_priors = {}
        self.per_task_class_num = args.num_classes // args.num_tasks

    def _map_global_to_local_label(self, global_label, actual_task_id):
        """
        将全局标签映射为本地标签
        global_label: 数据集中的原始标签
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID）
        映射公式：local_label = global_label - actual_task_id * per_task_class_num
        映射后标签范围：[0, per_task_class_num)
        """
        return global_label - actual_task_id * self.per_task_class_num

    def _compute_class_priors(self, loader, classes, task_id):
        """计算每个类别的先验概率 π_i^y = n_i^y / n_i + δ"""
        class_counts = {c: 0 for c in classes}
        total_samples = 0

        for _, y in loader:
            for label in y:
                class_counts[label.item()] += 1
                total_samples += 1

        delta = 1e-6
        priors = torch.ones(len(classes), device=self.args.device) * delta
        for i, cls in enumerate(classes):
            priors[i] += class_counts[cls] / total_samples

        self.class_priors[task_id] = priors
        return priors

    def calibrated_cross_entropy(self, logits, targets, priors):
        """
        校准的交叉熵损失 (CCE Loss)
        ℓ_cce(f_θ(̃x_ij), y_ij, π_i) = -log σ^{y_ij}(f_θ(̃x_ij) + log π_i)
        """
        calibrated_logits = logits + torch.log(priors + 1e-10)
        return F.cross_entropy(calibrated_logits, targets)

    def calibrated_kl_divergence(self, adv_logits, clean_logits, priors):
        """
        校准的KL散度损失 (CKL Loss)
        ℓ_ckl = -∑ σ^y(f_θ(x_ij) + log π_i) log σ^y(f_θ(x'_ij) + log π_i)
        """
        calibrated_clean = clean_logits + torch.log(priors + 1e-10)
        calibrated_adv = adv_logits + torch.log(priors + 1e-10)

        softmax_clean = F.softmax(calibrated_clean, dim=1)
        log_softmax_adv = F.log_softmax(calibrated_adv, dim=1)

        kl_div = -torch.sum(softmax_clean * log_softmax_adv, dim=1)
        return kl_div.mean()

    def pgd_attack_calfat(self, x, y, task_id, priors, eps=0.3, alpha=0.01, iters=40):
        """
        使用CKL损失生成对抗样本 (CalFAT版本)
        ̃x_ij = argmax ℓ_ckl(f_θ_i(x'_ij), f_θ_i(x_ij), π_i)
        """
        adv_x = x.clone().detach()
        adv_x = adv_x + torch.empty_like(adv_x).uniform_(-eps, eps)
        adv_x = torch.clamp(adv_x, 0, 1)

        self.model.eval()

        with torch.no_grad():
            clean_logits = self.model(x, task_id)
            if isinstance(clean_logits, tuple):
                clean_logits = clean_logits[0]

        for _ in range(iters):
            adv_x.requires_grad = True
            adv_logits = self.model(adv_x, task_id)
            if isinstance(adv_logits, tuple):
                adv_logits = adv_logits[0]

            loss = self.calibrated_kl_divergence(adv_logits, clean_logits, priors)

            grad = torch.autograd.grad(loss, adv_x, retain_graph=False)[0]

            adv_x = adv_x.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_x - x, -eps, eps)
            adv_x = torch.clamp(x + delta, 0, 1)

        self.model.train()
        return adv_x

    def set_weights(self, global_weights):
        """同步全局权重"""
        if global_weights is not None:
            self.model.load_state_dict(global_weights)
    
    def set_causal_bn(self, global_causal_bn):
        """同步全局因果模型BN层参数"""
        if global_causal_bn is not None:
            model_state = self.model.state_dict()
            for key in global_causal_bn:
                if key in model_state:
                    model_state[key] = global_causal_bn[key]
            self.model.load_state_dict(model_state)

    def _get_adversarial_x(self, x, y, task_id,attack_name=None,steps=None):
        """
        统一攻击入口：显式分支判断 
        """
        name = attack_name if attack_name else self.args.attack

        # 根据名字显式调用 
        if name == 'pgd':
            return self.pgd(x, y, task_id,steps=steps)
        elif name == 'fgsm':
            return self.fgsm(x, y,task_id)
        else:
            raise ValueError(f"Unsupported attack: {name}")


    def fgsm(self, x, y,task_id):
        """Fast Gradient Sign Method (FGSM)"""
        training_mode = self.model.training
        self.model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = self.model(x_adv,task_id)
        # 模型返回元组 (logits, features)，只取 logits
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + self.args.fgsm_eps * grad.sign()
        self.model.train(training_mode)
        return torch.clamp(x_adv, 0, 1).detach()
    
    def pgd(self, x, y, task_id,steps=None):
        """Projected Gradient Descent (PGD)"""
        training_mode = self.model.training
        self.model.eval()

        # 确定迭代步数：默认使用测试步数，除非显式指定（如训练时）
        _steps = steps if steps is not None else self.args.pgd_steps

        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.args.pgd_eps, self.args.pgd_eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv,task_id)
            # 模型返回元组 (logits, features)，只取 logits
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + self.args.pgd_alpha * grad.sign()
                delta = torch.clamp(x_adv - x, -self.args.pgd_eps, self.args.pgd_eps)
                x_adv = torch.clamp(x + delta, 0, 1).detach()

        self.model.train(training_mode)
        return x_adv


    def train_main_task(self, seq_idx, global_weights):
        """标准本地训练 (Clean Training)
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_task_id)

                # 模型输出选择分类器分支（基于全局任务ID）
                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, local_y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def pgd_train_main_task(self, seq_idx, global_weights):
        """
        对抗训练 (CalFAT Adversarial Training)
        使用CKL损失生成对抗样本，使用CCE损失进行训练
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        priors = self._compute_class_priors(loader, classes, actual_task_id)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_task_id)

                # 使用全局任务ID生成对抗样本（选择正确的分类器分支）
                x_adv = self.pgd_attack_calfat(
                    x, local_y, seq_idx, priors,
                    eps=self.args.pgd_eps if hasattr(self.args, 'pgd_eps') else 0.3,
                    alpha=self.args.pgd_alpha if hasattr(self.args, 'pgd_alpha') else 0.01,
                    iters=self.args.pgd_steps if hasattr(self.args, 'pgd_steps') else 40
                )

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.calibrated_cross_entropy(outputs, local_y, priors)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples
    
    def _compute_causal_loss(self, F_adv: torch.Tensor, F_natural: torch.Tensor) -> torch.Tensor:
        """
        计算因果损失（广义矩估计）
        
        目标：让因果模型从对抗特征中恢复出干净特征
        因果模型输入输出维度相同（都是特征维度）
        
        Args:
            F_adv: 对抗特征, shape [batch, feature_dim]
            F_natural: 干净特征, shape [batch, feature_dim]
        
        Returns:
            loss_causal: 因果损失
            Y_cau: 因果模型预测, shape [batch, feature_dim]
        """
        
        # 1. 构造工具变量 Z = F_adv - F_natural
        Z = F_adv - F_natural.detach()
        
        # 2. 因果模型预测（输入输出维度相同）
        Y_cau = self.model.base_model.causal_model(F_adv, Z)
        
        # 3. 矩条件损失
        # 目标是让因果模型恢复干净特征
        residual = F_natural - Y_cau
        
        # 矩条件: E[Z * residual] = 0
        # 使用工具变量Z作为测试函数
        # 添加数值稳定性处理：对 Z 和 residual 进行归一化
        Z_norm = Z / (torch.norm(Z, dim=1, keepdim=True) + 1e-10)
        residual_norm = residual / (torch.norm(residual, dim=1, keepdim=True) + 1e-10)
        
        moment = (Z_norm * residual_norm).mean()
        loss_causal = torch.abs(moment)
        
        # 添加更强的 L2 正则化防止因果模型输出过大
        loss_causal = loss_causal + 1e-3 * torch.norm(Y_cau)
        
        # 对因果损失进行裁剪，防止爆炸
        loss_causal = torch.clamp(loss_causal, min=0, max=5.0)
        
        return loss_causal, Y_cau
    
    def fcat_main_train_Cal(self, seq_idx):
        """
        Fcat 本地训练：结合因果模型和对抗训练
        根据注释中的伪代码实现
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        actual_task_id = self.task_sequence[seq_idx]
        
        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        # 使用更小的学习率进行对抗训练
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr * 0.1)
        
        # 计算类先验概率，用于 CCE 损失
        priors = self._compute_class_priors(loader, classes, actual_task_id)
        
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0
        total_adv_loss = 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_task_id)

                # 1. 提取自然特征和对抗特征
                F_natural = self.model.get_backbone_features(x)
                # 使用 pgd_attack_calfat 生成对抗样本
                x_adv=self._get_adversarial_x(x, local_y, seq_idx, attack_name='pgd',steps=self.args.pgd_steps)
                F_adv = self.model.get_backbone_features(x_adv)

                # 2. 计算因果损失（因果模型输入输出维度相同）
                loss_cau, Y_cau_pred = self._compute_causal_loss(F_adv, F_natural)

                # 3. 计算对抗训练损失（使用 CCE 损失）
                outputs_adv, _ = self.model(x_adv, seq_idx)
                loss_adv = self.calibrated_cross_entropy(outputs_adv, local_y, priors)
                
                # 对对抗损失进行裁剪，防止爆炸
                loss_adv = torch.clamp(loss_adv, min=0, max=20.0)

                # 4. 因果正则化 (KL散度)
                Z = F_adv - F_natural.detach()  # 重新计算工具变量用于因果正则化
                outputs_cau, _ = self.model(x_adv, seq_idx, use_causal=True, F_adv=F_adv, Z=Z)
                F_causal_out = outputs_cau
                F_adv_out = outputs_adv
                # 修复 KL 散度：F.kl_div 的第一个参数应该是 log(probs)，第二个是 probs
                # 添加数值稳定性处理
                eps = 1e-10
                log_probs_cau = F.log_softmax(F_causal_out, dim=1).clamp(min=-100, max=0)
                probs_adv = F.softmax(F_adv_out, dim=1).clamp(min=eps, max=1-eps)
                loss_kl = F.kl_div(log_probs_cau, probs_adv, reduction='batchmean')
                # 对 KL 损失进行更严格的裁剪
                loss_kl = torch.clamp(loss_kl, min=0, max=5.0)

                # 5. 总损失反向传播
                total_loss = loss_adv + lambda1 * loss_cau + lambda2 * loss_kl
                
                # 检查损失是否异常
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"[Client {self.cid}] Warning: Loss is NaN or Inf! Skipping batch...")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                
                # 更强的梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()

                with torch.no_grad():
                    outputs, _ = self.model(x, seq_idx)
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += total_loss.item()
                    total_adv_loss += loss_adv.item()

        # 提取因果模型的BN层参数上传
        bn_params = {}
        model_state = self.model.state_dict()
        for key in model_state:
            if 'causal_model' in key and ('bn' in key.lower() or 'batchnorm' in key.lower()):
                bn_params[key] = model_state[key]

        return bn_params, epoch_loss / total_samples, total_correct / total_samples, total_adv_loss / total_samples

    def fcat_main_train(self, seq_idx):
        """
        Fcat 本地训练：结合因果模型和对抗训练
        根据注释中的伪代码实现
        """
        actual_task_id = self.task_sequence[seq_idx]
        
        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        num_classes_per_task = len(classes)

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0
        total_adv_loss = 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1:
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_task_id)

                # 1. 提取自然特征和对抗特征
                F_natural = self.model.get_backbone_features(x)
                x_adv = self._get_adversarial_x(x, y_map, seq_idx,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)
                F_adv = self.model.get_backbone_features(x_adv)

                # 2. 计算因果损失（因果模型输入输出维度相同）
                loss_cau, Y_cau_pred = self._compute_causal_loss(F_adv, F_natural)

                # 3. 计算对抗训练损失
                outputs_adv, _ = self.model(x_adv, seq_idx)
                loss_adv = criterion(outputs_adv, y_map)

                # 4. 因果正则化 (KL散度)
                Z = F_adv - F_natural.detach()  # 重新计算工具变量用于因果正则化
                outputs_cau, _ = self.model(x_adv, seq_idx, use_causal=True, F_adv=F_adv, Z=Z)
                F_causal_out = outputs_cau
                F_adv_out = outputs_adv
                # 修复 KL 散度：F.kl_div 的第一个参数应该是 log(probs)，第二个是 probs
                loss_kl = F.kl_div(F.log_softmax(F_causal_out, dim=1), 
                                  F.softmax(F_adv_out, dim=1), reduction='batchmean')
                # 添加数值稳定性，防止 KL 散度爆炸
                loss_kl = torch.clamp(loss_kl, min=0, max=10.0)

                # 5. 总损失反向传播
                total_loss = loss_adv + lambda1 * loss_cau + lambda2 * loss_kl
                optimizer.zero_grad()
                total_loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    outputs, _ = self.model(x, seq_idx)
                    total_correct += (outputs.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += total_loss.item()
                    total_adv_loss += loss_adv.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        # 提取因果模型的BN层参数上传
        bn_params = {}
        model_state = self.model.state_dict()
        for key in model_state:
            if 'causal_model' in key and ('bn' in key.lower() or 'batchnorm' in key.lower()):
                bn_params[key] = model_state[key]

        return bn_params, epoch_loss / total_samples, total_correct / total_samples, total_adv_loss / total_samples, training_time, mem_start, mem_peak, mem_end
    def extensive_test(self, curr_round):
        """全面测试：评估 Clean 和 Robust 准确率
        s_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_tid: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        max_seq_idx = min(curr_round // self.args.t_round, self.args.num_tasks - 1)
        results = {'clean': [], 'robust': []}

        self.model.eval()
        for s_idx in range(max_seq_idx + 1):
            actual_tid = self.task_sequence[s_idx]
            loader, classes = self.data_generator.get_loader(self.cid, actual_tid, 'test')

            correct_clean, correct_adv, total = 0, 0, 0

            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_tid)

                # 使用全局任务ID进行测试（选择正确的分类器分支）
                with torch.no_grad():
                    out_clean, _ = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == local_y).sum().item()

                # 使用全局任务ID生成对抗样本并测试
                x_adv = self._get_adversarial_x(x, local_y, s_idx, steps=self.args.pgd_test)

                with torch.no_grad():
                    out_adv, _ = self.model(x_adv, s_idx)
                    correct_adv += (out_adv.argmax(1) == local_y).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            results['robust'].append(correct_adv / total if total > 0 else 0)

        return results