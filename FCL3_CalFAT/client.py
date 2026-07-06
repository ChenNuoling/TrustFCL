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
        return global_label - actual_task_id * self.per_task_class_num

    def set_weights(self, global_weights):
        """同步全局权重"""
        if global_weights is not None:
            self.model.load_state_dict(global_weights)

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
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + self.args.pgd_alpha * grad.sign()
                delta = torch.clamp(x_adv - x, -self.args.pgd_eps, self.args.pgd_eps)
                x_adv = torch.clamp(x + delta, 0, 1).detach()

        self.model.train(training_mode)
        return x_adv

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

        for _ in range(iters):
            adv_x.requires_grad = True
            adv_logits = self.model(adv_x, task_id)

            loss = self.calibrated_kl_divergence(adv_logits, clean_logits, priors)

            grad = torch.autograd.grad(loss, adv_x, retain_graph=False)[0]

            adv_x = adv_x.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_x - x, -eps, eps)
            adv_x = torch.clamp(x + delta, 0, 1)

        self.model.train()
        return adv_x


    def train_main_task(self, seq_idx, global_weights):
        """标准本地训练 (Clean Training)
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）- 用于选择分类器分支
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）- 用于标签映射
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = self._map_global_to_local_label(y, actual_task_id)

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                loss = criterion(outputs, local_y)
                loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end

    def calfat_train_main_task(self, seq_idx, global_weights):
        """
        CalFAT训练方法：
        1. 计算类别先验概率 π_i
        2. 使用CKL损失生成对抗样本
        3. 使用CCE损失进行优化
        
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）- 用于选择分类器分支
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）- 用于标签映射
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        # 计算类别先验概率
        priors = self._compute_class_priors(loader, classes, actual_task_id)

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = self._map_global_to_local_label(y, actual_task_id)

                # 使用CKL损失生成对抗样本（使用全局任务ID选择分类器分支）
                x_adv = self.pgd_attack_calfat(
                    x, local_y, seq_idx, priors,
                    eps=self.args.pgd_eps if hasattr(self.args, 'pgd_eps') else 0.3,
                    alpha=self.args.pgd_alpha if hasattr(self.args, 'pgd_alpha') else 0.01,
                    iters=self.args.pgd_steps if hasattr(self.args, 'pgd_steps') else 40
                )

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)

                # 使用CCE损失进行优化
                loss = self.calibrated_cross_entropy(outputs, local_y, priors)

                loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end

    def extensive_test(self, curr_round):
        """全面测试：评估 Clean 和 Robust 准确率
        s_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）- 用于选择分类器分支
        actual_tid: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）- 用于标签映射
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

                local_y = self._map_global_to_local_label(y, actual_tid)

                with torch.no_grad():
                    out_clean = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == local_y).sum().item()

                # 测试时根据 args.attack 动态生成对抗样本（使用全局任务ID选择分类器分支）
                x_adv = self._get_adversarial_x(x, local_y, s_idx, steps=self.args.pgd_test)

                with torch.no_grad():
                    out_adv = self.model(x_adv, s_idx)
                    correct_adv += (out_adv.argmax(1) == local_y).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            results['robust'].append(correct_adv / total if total > 0 else 0)

        return results