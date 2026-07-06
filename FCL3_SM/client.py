import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from utils import compute_class_prior, compute_energy_normalized_saliency, get_memory_usage



class Client:
    def __init__(self, cid, args, model, data_loader, task_sequence):
        self.cid = cid
        self.args = args
        self.model = model.to(args.device)
        self.data_generator = data_loader
        self.task_sequence = task_sequence
        self.sm_agg = None
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
    
    def set_weights(self, global_weights):
        """同步全局权重"""
        if global_weights is not None:
            self.model.load_state_dict(global_weights)
    
    def set_sm_agg(self, sm_agg):
        """设置服务器下发的SM_agg"""
        self.sm_agg = sm_agg

    def _compute_calfat_ce_loss(self, logits, y, class_prior):
        """
        CalFAT Class-Calibrated Cross Entropy Loss
        参数:
            logits: 模型输出的未归一化 logits, shape (batch_size, num_classes)
            y: 真实标签, shape (batch_size,)
            class_prior: 类先验概率, shape (num_classes,)
        返回:
            loss: 标量损失值
        """
        eps = 1e-10
        log_prior = torch.log(class_prior + eps)
        # logits + log(prior) 进行对数校准
        calibrated_logits = logits + log_prior
        # 计算交叉熵损失
        loss = F.cross_entropy(calibrated_logits, y)
        return loss
    
    def _compute_ckl_loss(self, logits_clean, logits_adv):
        """
        Contrastive KL Divergence Loss
        参数:
            logits_clean: 原始样本的logits
            logits_adv: 对抗样本的logits
        返回:
            loss: -CKL (因为要最大化CKL)
        """
        p_clean = F.softmax(logits_clean, dim=1)
        p_adv = F.softmax(logits_adv, dim=1)
        # 计算 KL(p_adv || p_clean)
        kl_div = F.kl_div(F.log_softmax(logits_adv, dim=1), p_clean, reduction='batchmean')
        # 最大化CKL等价于最小化-CKL
        return -kl_div
    
    def _compute_enhanced_saliency_map(self, x, task_id, target_y):
        """
        生成增强型显著性图 S = (s'_r, s'_g, s'_b)
        参数:
            x: 输入图像, shape (batch_size, 3, H, W)
            task_id: 任务ID
            target_y: 目标标签, shape (batch_size,)
        返回:
            S_enhanced: 增强后的显著性图, shape (3, H, W)
        """
        training_mode = self.model.training
        self.model.eval()
        x.requires_grad_(True)
        
        # 前向传播计算梯度
        outputs = self.model(x, task_id)
        loss = F.cross_entropy(outputs, target_y)
        loss.backward()
        
        # 获取输入梯度 (shape: (B, 3, H, W))
        grad = x.grad.detach()
        
        # 对每个通道分别归一化
        batch_sal_maps = []
        for i in range(x.size(0)):
            # 取绝对值
            abs_grad = torch.abs(grad[i])
            # 能量归一化
            normalized_sal = compute_energy_normalized_saliency(abs_grad)
            batch_sal_maps.append(normalized_sal)
        
        # 取batch平均值
        sal_map = torch.stack(batch_sal_maps).mean(dim=0)
        
        x.grad.zero_()
        self.model.train(training_mode)
        
        return sal_map
    
    def _compute_task_saliency_map(self, task_id):
        """
        计算当前任务所有图像的平均SM
        参数:
            task_id: 任务ID
        返回:
            task_sm: 平均SM图, shape (3, H, W)
        """
        actual_task_id = self.task_sequence[task_id]
        training_mode = self.model.training
        self.model.eval()
        
        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}
        
        accumulated_sm = None
        count = 0
        
        for x, y in loader:
            if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
            x, y = x.to(self.args.device), y.to(self.args.device)
            y_map = self._map_global_to_local_label(y, actual_task_id)
            
            # 计算当前batch的SM
            batch_sm = self._compute_enhanced_saliency_map(x, task_id, y_map)
            
            if accumulated_sm is None:
                accumulated_sm = batch_sm
            else:
                accumulated_sm += batch_sm
            count += 1
        
        # 计算平均值
        if count > 0:
            task_sm = accumulated_sm / count
        else:
            task_sm = torch.zeros(3, 32, 32, device=self.args.device)
        
        self.model.train(training_mode)
        return task_sm
    
    def _fuse_saliency_maps(self, sm_old, sm_new):
        """
        取新旧SM的逐像素最大值
        参数:
            sm_old: 旧SM (训练前), shape (3, H, W)
            sm_new: 新SM (训练后), shape (3, H, W)
        返回:
            fused_sm: 融合后的SM, shape (3, H, W)
        """
        return torch.max(sm_old, sm_new)
    
    def _compute_saliency_distillation_loss(self, sm_t, sm_agg, sm_prev=None):
        """
        SM_v3 SD-Loss: 当SM_t < SM_agg 或 SM_t < S_t-1时惩罚
        参数:
            sm_t: 当前SM, shape (3, H, W)
            sm_agg: 聚合后的SM_agg, shape (3, H, W)
            sm_prev: S_t-1, 上一任务的SM, shape (3, H, W) (可选)
        返回:
            loss: SD-Loss值
        """
        if sm_prev is not None:
            target_sm = torch.max(sm_agg, sm_prev)
        else:
            target_sm = sm_agg
        
        diff = torch.max(target_sm - sm_t, torch.zeros_like(sm_t))
        loss = torch.mean(diff ** 2)
        return loss

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


    def train_main_task(self, seq_idx, global_weights):
        """标准本地训练 (Clean Training)"""
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_task_id)

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                loss = criterion(outputs, y_map)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples if total_samples > 0 else 0, total_correct / total_samples if total_samples > 0 else 0

    def train_main_task_with_calfat(self, seq_idx, sm_agg=None):
        """
        带CalFAT损失和SM引导的对抗训练
        参数:
            seq_idx: 任务在序列中的索引
            sm_agg: 服务器下发的SM_agg (可选，优先使用self.sm_agg)
        返回:
            state_dict, loss, accuracy, task_sm (任务SM), training_time, mem_start, mem_peak, mem_end
        """
        actual_task_id = self.task_sequence[seq_idx]
        
        # 优先使用self.sm_agg（由set_sm_agg设置），否则使用传入的参数
        sm_agg_to_use = self.sm_agg if self.sm_agg is not None else sm_agg

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}
        
        # 收集所有标签以计算类先验
        all_labels = []
        for _, y in loader:
            all_labels.extend([label_map[l.item()] for l in y])
        all_labels = torch.tensor(all_labels, device=self.args.device)
        num_classes = len(classes)
        class_prior = compute_class_prior(all_labels, num_classes)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()
        
        # 【训练前】计算 SM_before (旧任务SM)
        sm_before = self._compute_task_saliency_map(seq_idx)
        
        epoch_loss, total_correct, total_samples = 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_task_id)
                
                # 生成对抗样本
                x_adv = self._get_adversarial_x(x, y_map, seq_idx, 
                                                attack_name='pgd', steps=self.args.pgd_steps)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits_clean = self.model(x, seq_idx)
                logits_adv = self.model(x_adv, seq_idx)
                
                # 计算CalFAT损失
                loss_ce = self._compute_calfat_ce_loss(logits_adv, y_map, class_prior)
                loss_ckl = self._compute_ckl_loss(logits_clean, logits_adv)
                
                # 总损失
                total_loss = loss_ce + self.args.ckl_lambda * loss_ckl
                
                # 若提供sm_agg，计算SD-Loss (SM_v3: 同时考虑SM_agg和sm_before)
                if sm_agg_to_use is not None:
                    sm_current = self._compute_enhanced_saliency_map(x, seq_idx, y_map)
                    loss_sd = self._compute_saliency_distillation_loss(sm_current, sm_agg_to_use, sm_before)
                    total_loss += self.args.sm_lambda * loss_sd
                
                total_loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (logits_adv.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += total_loss.item()
        
        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        # 【训练后】计算 SM_after (新SM)
        sm_after = self._compute_task_saliency_map(seq_idx)
        
        # 融合新旧SM
        task_sm = self._fuse_saliency_maps(sm_before, sm_after)
        
        return (self.model.state_dict(), 
                epoch_loss / total_samples if total_samples > 0 else 0, 
                total_correct / total_samples if total_samples > 0 else 0,
                task_sm,
                training_time, mem_start, mem_peak, mem_end)

    def pgd_train_main_task(self, seq_idx, global_weights):
        """
        对抗训练 (PGD Adversarial Training) ❗
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_task_id)

                # 生成对抗样本 (固定使用 pgd)
                x_adv = self._get_adversarial_x(x, y_map,seq_idx,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv,seq_idx)
                loss = criterion(outputs, y_map)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples if total_samples > 0 else 0, total_correct / total_samples if total_samples > 0 else 0

    def extensive_test(self, curr_round):
        """全面测试：评估 Clean 和 Robust 准确率"""
        max_seq_idx = min(curr_round // self.args.t_round, self.args.num_tasks - 1)
        results = {'clean': [], 'robust': []}

        self.model.eval()
        for s_idx in range(max_seq_idx + 1):
            actual_tid = self.task_sequence[s_idx]
            loader, classes = self.data_generator.get_loader(self.cid, actual_tid, 'test')
            label_map = {g: l for l, g in enumerate(classes)}

            correct_clean, correct_adv, total = 0, 0, 0

            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_tid)

                with torch.no_grad():
                    out_clean = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == y_map).sum().item()

                # 测试时根据 args.attack 动态生成对抗样本
                x_adv = self._get_adversarial_x(x, y_map, s_idx,steps=self.args.pgd_test)

                with torch.no_grad():
                    out_adv = self.model(x_adv, s_idx)
                    correct_adv += (out_adv.argmax(1) == y_map).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            results['robust'].append(correct_adv / total if total > 0 else 0)

        return results