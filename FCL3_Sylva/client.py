import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import time
from utils import get_memory_usage



class Client:
    def __init__(self, cid, args, model, data_loader, task_sequence):
        self.cid = cid
        self.args = args
        self.model = model.to(args.device)
        self.data_generator = data_loader
        self.task_sequence = task_sequence
        self.per_task_class_num = args.num_classes // args.num_tasks
        self.class_priors = {}

    def _map_global_to_local_label(self, global_label, actual_task_id):
        """
        将全局标签映射为本地标签
        global_label: 数据集中的原始标签
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID）
        映射公式：local_label = global_label - actual_task_id * per_task_class_num
        映射后标签范围：[0, per_task_class_num)
        """
        return global_label - actual_task_id * self.per_task_class_num

    def set_weights(self, global_weights):
        """同步全局权重"""
        if global_weights is not None:
            self.model.load_state_dict(global_weights)

    def update_lora_params(self, global_lora):
        """只更新模型中的LoRA参数"""
        if global_lora is None:
            return
        current_state = self.model.state_dict()
        current_state.update(global_lora)
        self.model.load_state_dict(current_state)

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

    def _compute_class_priors(self, loader, actual_task_id):
        """
        计算每个类别的先验概率
        π_i = n_i / n + δ
        """
        class_counts = {l: 0 for l in range(self.per_task_class_num)}
        total_samples = 0

        for _, y in loader:
            local_y = self._map_global_to_local_label(y, actual_task_id)
            for label in local_y:
                class_counts[label.item()] += 1
                total_samples += 1

        delta = 1e-6
        priors = torch.ones(self.per_task_class_num, device=self.args.device) * delta
        for i in range(self.per_task_class_num):
            priors[i] += class_counts[i] / total_samples if total_samples > 0 else 0

        self.class_priors[actual_task_id] = priors
        return priors

    def calibrated_cross_entropy(self, logits, targets, priors):
        """
        校准的交叉熵损失 (CCE Loss)
        ℓ_cce = -log σ^{y}(f(x) + log π)
        """
        calibrated_logits = logits + torch.log(priors + 1e-10)
        return F.cross_entropy(calibrated_logits, targets)

    def calibrated_kl_divergence(self, adv_logits, clean_logits, priors):
        """
        校准的KL散度损失 (CKL Loss)
        ℓ_ckl = -∑ σ^y(f(x) + log π) log σ^y(f(x') + log π)
        """
        calibrated_clean = clean_logits + torch.log(priors + 1e-10)
        calibrated_adv = adv_logits + torch.log(priors + 1e-10)

        softmax_clean = F.softmax(calibrated_clean, dim=1)
        log_softmax_adv = F.log_softmax(calibrated_adv, dim=1)

        kl_div = -torch.sum(softmax_clean * log_softmax_adv, dim=1)
        return kl_div.mean()

    def pgd_attack_calfat(self, x, y, seq_idx, priors, eps=0.3, alpha=0.01, iters=40):
        """
        使用CKL损失生成对抗样本 (CalFAT版本)
        ̃x = argmax ℓ_ckl(f(x'), f(x), π)
        """
        adv_x = x.clone().detach()
        adv_x = adv_x + torch.empty_like(adv_x).uniform_(-eps, eps)
        adv_x = torch.clamp(adv_x, 0, 1)

        self.model.eval() 

        with torch.no_grad():
            clean_logits = self.model(x, seq_idx)

        for _ in range(iters):
            adv_x.requires_grad = True
            adv_logits = self.model(adv_x, seq_idx)

            loss = self.calibrated_kl_divergence(adv_logits, clean_logits, priors)

            grad = torch.autograd.grad(loss, adv_x, retain_graph=False)[0]

            adv_x = adv_x.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_x - x, -eps, eps)
            adv_x = torch.clamp(x + delta, 0, 1)

        self.model.train()
        return adv_x


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
        对抗训练 (PGD Adversarial Training)
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

                # 使用本地标签生成对抗样本（模型输出基于全局任务ID）
                x_adv = self._get_adversarial_x(x, local_y, seq_idx,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)
                loss = criterion(outputs, local_y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def pretrain_backbone(self, seq_idx, epochs=50):
        """
        预训练主干网络：只训练主干模型和分类器，冻结LoRA
        在每个任务的第一个通信轮次调用一次，训练到收敛
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        actual_task_id = self.task_sequence[seq_idx]

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        # 只训练主干模型和分类器，冻结LoRA
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = optim.Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.args.lr
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss, total_correct, total_samples = 0, 0, 0
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_task_id)

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                loss = criterion(outputs, local_y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                acc = total_correct / total_samples if total_samples > 0 else 0
                print(f"    Client {self.cid}: Pretrain Backbone Epoch {epoch+1}/{epochs}, Loss={epoch_loss/total_samples:.4f}, Acc={acc:.2f}")

        # 恢复所有参数的requires_grad
        for param in self.model.parameters():
            param.requires_grad = True

    def _get_lora_params(self, model):
        """获取模型中所有LoRA参数"""
        lora_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_params[name] = param.data.clone()
        return lora_params

    def sylva_train_main_task_Cal(self, seq_idx, global_weights=None):
        """
        Sylva本地对抗训练（CalFAT版本）：
        - 只训练LoRA和分类器（模型参数已在聚合后更新）
        - 损失函数：校准交叉熵(CCE) + 校准KL散度(CKL) + L2正则化
        - 只传回LoRA参数
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        actual_task_id = self.task_sequence[seq_idx]
        if global_weights is not None:
            self.set_weights(global_weights)
        # 保存当前LoRA状态，用于L2正则化
        global_lora_state = copy.deepcopy(self._get_lora_params(self.model))

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        # 计算类别先验概率
        priors = self._compute_class_priors(loader, actual_task_id)

        # 只训练LoRA参数和分类器
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad], lr=self.args.lr)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                # 将全局标签映射为本地标签（基于本地真实任务ID）
                local_y = self._map_global_to_local_label(y, actual_task_id)

                # 使用CKL损失生成对抗样本（CalFAT版本）
                x_adv = self.pgd_attack_calfat(
                    x, local_y, seq_idx, priors,
                    eps=self.args.pgd_eps if hasattr(self.args, 'pgd_eps') else 0.3,
                    alpha=self.args.pgd_alpha if hasattr(self.args, 'pgd_alpha') else 0.01,
                    iters=self.args.pgd_steps if hasattr(self.args, 'pgd_steps') else 40
                )

                optimizer.zero_grad()

                # 干净样本前向（模型输出基于全局任务ID）
                logits_clean = self.model(x, seq_idx)
                # 对抗样本前向（模型输出基于全局任务ID）
                logits_adv = self.model(x_adv, seq_idx)

                # 1. 校准交叉熵损失 (CCE Loss)
                loss_ce = self.calibrated_cross_entropy(logits_adv, local_y, priors)

                # 2. 校准KL散度损失 (CKL Loss)
                loss_kl = self.calibrated_kl_divergence(logits_adv, logits_clean, priors)

                # 3. L2正则化：LoRA参数不要偏离全局模型太远
                loss_l2 = 0.0
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        loss_l2 += torch.sum((param - global_lora_state[name])**2)

                # 总损失
                loss = loss_ce + self.args.sylva_kl_weight * loss_kl + self.args.sylva_l2_weight * loss_l2

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (logits_adv.argmax(dim=1) == local_y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        # 恢复所有参数的requires_grad
        for param in self.model.parameters():
            param.requires_grad = True

        # 只传回LoRA参数
        lora_params = self._get_lora_params(self.model)
        return lora_params, epoch_loss / total_samples, total_correct / total_samples

    def sylva_train_main_task(self, seq_idx, global_weights=None):
        """
        Sylva本地对抗训练：
        - 只训练LoRA和分类器（模型参数已在聚合后更新）
        - 损失函数：加权交叉熵 + KL散度 + L2正则化
        - 只传回LoRA参数
        """
        actual_task_id = self.task_sequence[seq_idx]
        # 保存当前LoRA状态，用于L2正则化
        global_lora_state = copy.deepcopy(self._get_lora_params(self.model))

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}
        
        # 统计类别权重
        class_counts = {l: 0 for l in label_map.values()}
        for _, y in loader:
            y_map = torch.tensor([label_map[li.item()] for li in y], device=self.args.device)
            for yi in y_map:
                class_counts[yi.item()] = class_counts.get(yi.item(), 0) + 1
        total = sum(class_counts.values())
        class_weights = torch.tensor([total / max(count, 1) for count in class_counts.values()], device=self.args.device)

        # 只训练LoRA参数和分类器
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad], lr=self.args.lr)
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1:
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = torch.tensor([label_map[l.item()] for l in y], device=self.args.device)

                # 生成对抗样本
                x_adv = self._get_adversarial_x(x, y_map, actual_task_id,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)

                optimizer.zero_grad()
                
                # 干净样本前向
                logits_clean = self.model(x, actual_task_id)
                # 对抗样本前向
                logits_adv = self.model(x_adv, actual_task_id)
                
                # 1. 加权交叉熵损失
                loss_ce = ce_criterion(logits_adv, y_map)
                
                # 2. KL散度损失：保持干净样本和对抗样本输出的一致性
                prob_clean = F.softmax(logits_clean, dim=1)
                prob_adv = F.softmax(logits_adv, dim=1)
                loss_kl = F.kl_div(torch.log(prob_adv + 1e-10), prob_clean, reduction='batchmean')
                
                # 3. L2正则化：LoRA参数不要偏离全局模型太远
                loss_l2 = 0.0
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        loss_l2 += torch.sum((param - global_lora_state[name])**2)
                
                # 总损失
                loss = loss_ce + self.args.sylva_kl_weight * loss_kl + self.args.sylva_l2_weight * loss_l2
                
                loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (logits_adv.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        # 恢复所有参数的requires_grad
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 只传回LoRA参数
        lora_params = self._get_lora_params(self.model)
        return lora_params, epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end
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

                # 使用本地标签进行测试（模型输出基于全局任务ID）
                with torch.no_grad():
                    out_clean = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == local_y).sum().item()

                # 使用本地标签生成对抗样本并测试
                x_adv = self._get_adversarial_x(x, local_y, s_idx, steps=self.args.pgd_test)

                with torch.no_grad():
                    out_adv = self.model(x_adv, s_idx)
                    correct_adv += (out_adv.argmax(1) == local_y).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            results['robust'].append(correct_adv / total if total > 0 else 0)

        return results

    def _get_layer_names(self):
        """获取模型中所有可训练层的名称"""
        layer_names = []
        for name, _ in self.model.named_parameters():
            if 'base_model' in name:
                clean_name = name.replace('base_model.', '')
                layer_names.append(clean_name)
        return layer_names

    def _compute_shapley_values(self, seq_idx, actual_task_id, beta=0.5, B=300, p=0.03):
        """
        使用蒙特卡洛采样计算各层的Shapley值
        Shapley值 = 干净损失价值 - β * 鲁棒性损失价值
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        layer_names = self._get_layer_names()
        num_layers = len(layer_names)
        num_selected = max(1, int(num_layers * p))

        shapley_values = {name: 0.0 for name in layer_names}

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        if len(loader) == 0:
            return layer_names[:num_selected]

        clean_loader_iter = iter(loader)

        for b in range(B):
            try:
                x, y = next(clean_loader_iter)
            except StopIteration:
                clean_loader_iter = iter(loader)
                x, y = next(clean_loader_iter)

            x, y = x.to(self.args.device), y.to(self.args.device)

            # 将全局标签映射为本地标签（基于本地真实任务ID）
            local_y = self._map_global_to_local_label(y, actual_task_id)

            x_adv = self.pgd(x, local_y, seq_idx, steps=self.args.pgd_steps)

            permutation = np.random.permutation(num_layers)
            current_set = set()

            original_state = {name: param.clone() for name, param in self.model.named_parameters() if 'base_model' in name}

            for idx in permutation:
                layer_name = layer_names[idx]

                v_S = self._evaluate_subset_loss(current_set, x, local_y, x_adv, seq_idx)

                current_set.add(layer_name)
                v_S_with = self._evaluate_subset_loss(current_set, x, local_y, x_adv, seq_idx)

                marginal_contribution = (v_S_with - v_S) - beta * (self._evaluate_subset_loss(current_set, x, local_y, x_adv, seq_idx, use_adv=True) - self._evaluate_subset_loss(current_set, x, local_y, x_adv, seq_idx, use_adv=False))
                shapley_values[layer_name] += marginal_contribution / B

                for name, param in self.model.named_parameters():
                    if f'base_model.{layer_name}' == name or name == layer_name:
                        param.data = original_state.get(name, param).clone()
                        break

        sorted_layers = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
        selected_layers = [name for name, _ in sorted_layers[:num_selected]]

        return selected_layers

    def _evaluate_subset_loss(self, layer_set, x, y, x_adv, seq_idx, use_adv=False):
        """评估在特定层子集下模型的损失"""
        self.model.eval()

        train_params = set()
        for name, param in self.model.named_parameters():
            if 'base_model' in name:
                clean_name = name.replace('base_model.', '')
                if clean_name in layer_set:
                    train_params.add(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        eval_x = x_adv if use_adv else x

        outputs = self.model(eval_x, seq_idx)
        loss = criterion(outputs, y)

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        return loss.item()

    def Sylva_retrain(self, seq_idx):
        """
        二阶段训练：使用Shapley值选择需要训练的层，冻结其余层，
        仅用干净样本微调选中的层（模型参数已在聚合后更新）
        seq_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_task_id: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        返回: state_dict, loss, accuracy, retrain_time, mem_start, mem_peak, mem_end
        """
        actual_task_id = self.task_sequence[seq_idx]

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        selected_layers = self._compute_shapley_values(seq_idx, actual_task_id,
                                                      beta=self.args.sylva_beta,
                                                      B=self.args.sylva_B,
                                                      p=self.args.sylva_p)

        for name, param in self.model.named_parameters():
            if 'base_model' in name:
                clean_name = name.replace('base_model.', '')
                if clean_name in selected_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = optim.Adam(
            [param for name, param in self.model.named_parameters() if param.requires_grad],
            lr=self.args.lr * self.args.sylva_lr_ratio
        )
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

        retrain_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        for param in self.model.parameters():
            param.requires_grad = True

        return (self.model.state_dict(), 
                epoch_loss / total_samples if total_samples > 0 else 0, 
                total_correct / total_samples if total_samples > 0 else 0,
                retrain_time, mem_start, mem_peak, mem_end)