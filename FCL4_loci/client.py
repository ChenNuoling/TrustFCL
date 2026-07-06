import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from copy import deepcopy
import time
from nets import UnifiedKDModel
from utils import get_memory_usage


def MultiClassCrossEntropy(logits, labels, T=2):
    """数值稳定的MultiClassCrossEntropy"""
    # 数值稳定性保护
    logits = torch.clamp(logits, min=-1e5, max=1e5)
    labels = torch.clamp(labels, min=-1e5, max=1e5)
    
    outputs = torch.log_softmax(logits / T, dim=1)
    label = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    
    # 检查NaN
    if torch.isnan(outputs):
        outputs = torch.tensor(0.0, device=logits.device)
    
    return outputs


class Client:
    def __init__(self, cid, args, model, data_loader, task_sequence):
        self.cid = cid
        self.args = args
        self.model = model.to(args.device)
        self.data_generator = data_loader
        self.task_sequence = task_sequence
        self.fisher = None
        self.model_old = None
        self.lamb = getattr(args, 'ewc_lambda', 1000)
        self.kd_lambda = getattr(args, 'kd_lambda', 1.0)
        self.kd_epoch = getattr(args, 'kd_epoch', 5)
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.old_task = -1
        self.ce = nn.CrossEntropyLoss().to(args.device)
        self.kd_model = UnifiedKDModel(args).to(args.device)
        self.cur_kd = deepcopy(self.kd_model)
        self.first_train = True
        self.kd_models_history = []
        self.class_priors = {}

    def _compute_class_priors(self, loader, classes, task_id):
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
        calibrated_logits = logits + torch.log(priors + 1e-10)
        return F.cross_entropy(calibrated_logits, targets)

    def calibrated_kl_divergence(self, adv_logits, clean_logits, priors):
        calibrated_clean = clean_logits + torch.log(priors + 1e-10)
        calibrated_adv = adv_logits + torch.log(priors + 1e-10)

        softmax_clean = F.softmax(calibrated_clean, dim=1)
        log_softmax_adv = F.log_softmax(calibrated_adv, dim=1)

        kl_div = -torch.sum(softmax_clean * log_softmax_adv, dim=1)
        return kl_div.mean()

    def pgd_attack_calfat(self, x, y, task_id, priors, eps=0.3, alpha=0.01, iters=40):
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

    def set_weights(self, global_weights):
        """同步全局权重"""
        if global_weights is not None:
            self.model.load_state_dict(global_weights)

    def fisher_matrix_diag(self, seq_idx, actual_task_id, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            fisher[n] = 0 * p.data
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        all_num = 0
        
        per_task_class_num = self.args.num_classes // self.args.num_tasks
        
        for images, target in dataloader:
            images = images.to(self.args.device)
            target = target.to(self.args.device)
            
            local_target = target - actual_task_id * per_task_class_num
            
            all_num += images.shape[0]
            self.model.zero_grad()
            outputs = self.model.forward(images, seq_idx)
            loss = criterion(outputs, local_target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += images.shape[0] * p.grad.data.pow(2)
        with torch.no_grad():
            for n, _ in self.model.named_parameters():
                fisher[n] = fisher[n] / all_num
        return fisher

    def ewc_criterion(self, t, output, targets):
        loss_reg = 0
        if t > 0 and self.fisher is not None and self.model_old is not None:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg

    def _get_adversarial_x(self, x, y, task_id=None, attack_name=None, steps=None):
        """
        统一攻击入口：显式分支判断 
        """
        name = attack_name if attack_name else self.args.attack

        # 根据名字显式调用 
        if name == 'pgd':
            return self.pgd(x, y, task_id, steps=steps)
        elif name == 'fgsm':
            return self.fgsm(x, y, task_id)
        else:
            raise ValueError(f"Unsupported attack: {name}")


    def fgsm(self, x, y, task_id=None):
        """Fast Gradient Sign Method (FGSM)"""
        training_mode = self.model.training
        self.model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = self.model(x_adv, task_id)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + self.args.fgsm_eps * grad.sign()
        self.model.train(training_mode)
        return torch.clamp(x_adv, 0, 1).detach()
    
    def pgd(self, x, y, task_id=None, steps=None):
        """Projected Gradient Descent (PGD)"""
        training_mode = self.model.training
        self.model.eval()

        # 确定迭代步数：默认使用测试步数，除非显式指定（如训练时）
        _steps = steps if steps is not None else self.args.pgd_steps

        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.args.pgd_eps, self.args.pgd_eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv, task_id)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + self.args.pgd_alpha * grad.sign()
                delta = torch.clamp(x_adv - x, -self.args.pgd_eps, self.args.pgd_eps)
                x_adv = torch.clamp(x + delta, 0, 1).detach()

        self.model.train(training_mode)
        return x_adv


    def train_main_task(self, seq_idx, global_weights):
        """标准本地训练 (Clean Training) - 使用全局标签"""
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, _ = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == y).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def pgd_train_main_task(self, seq_idx, global_weights):
        """
        对抗训练 (CalFAT PGD Adversarial Training) - 使用CCE损失和CKL对抗样本
        """
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, _ = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        per_task_class_num = self.args.num_classes // self.args.num_tasks
        classes = list(range(per_task_class_num))

        if actual_task_id not in self.class_priors:
            self._compute_class_priors(loader, classes, actual_task_id)
        priors = self.class_priors[actual_task_id]

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = y - actual_task_id * per_task_class_num

                x_adv = self.pgd_attack_calfat(x, local_y, seq_idx, priors,
                                                eps=self.args.pgd_eps,
                                                alpha=self.args.pgd_alpha,
                                                iters=self.args.pgd_steps)

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)
                loss = self.calibrated_cross_entropy(outputs, local_y, priors)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += local_y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def train_ewc_task(self, seq_idx, global_weights):
        """EWC本地训练 (Elastic Weight Consolidation) - 使用全局标签映射"""
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, _ = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        if actual_task_id != self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            for param in self.model_old.parameters():
                param.requires_grad = False
            self.old_task = actual_task_id

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        per_task_class_num = self.args.num_classes // self.args.num_tasks

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = y - actual_task_id * per_task_class_num

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                loss = self.ewc_criterion(actual_task_id, outputs, local_y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += local_y.size(0)
                    epoch_loss += loss.item()

        if actual_task_id > 0:
            fisher_old = {}
            if self.fisher is not None:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher:
                        fisher_old[n] = self.fisher[n].clone()
            self.fisher = self.fisher_matrix_diag(seq_idx, actual_task_id, loader)
            if self.fisher is not None and actual_task_id > 0 and len(fisher_old) > 0:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher and n in fisher_old:
                        self.fisher[n] = (self.fisher[n] + fisher_old[n] * actual_task_id) / (actual_task_id + 1)

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def train_kd(self, actual_task_id, dataloader):
        """训练KD模型 (Knowledge Distillation) - 使用全局标签映射"""
        self.cur_kd.train()
        kd_optimizer = optim.Adam(self.cur_kd.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        per_task_class_num = self.args.num_classes // self.args.num_tasks
        seq_idx = self.task_sequence.index(actual_task_id) if actual_task_id in self.task_sequence else 0

        for epoch in range(self.kd_epoch):
            for images, targets in dataloader:
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                local_targets = targets - actual_task_id * per_task_class_num

                outputs = self.cur_kd.forward(images, seq_idx)
                loss = criterion(outputs, local_targets)
                kd_optimizer.zero_grad()
                loss.backward()
                kd_optimizer.step()

        return

    def train_ewc_kd_task(self, seq_idx, global_weights):
        """EWC+KD本地训练 (Elastic Weight Consolidation + Knowledge Distillation) - 使用全局标签映射"""
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, _ = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        if actual_task_id != self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            for param in self.model_old.parameters():
                param.requires_grad = False
            self.old_task = actual_task_id
            self.cur_kd = deepcopy(self.kd_model)
            self.first_train = True

        self.train_kd(actual_task_id, loader)
        self.cur_kd.eval()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        per_task_class_num = self.args.num_classes // self.args.num_tasks

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                optimizer.zero_grad()
                outputs = self.model(x, seq_idx)
                kd_outputs = self.cur_kd(x, seq_idx)

                local_y = y - actual_task_id * per_task_class_num

                loss_ewc = self.ewc_criterion(actual_task_id, outputs, local_y)
                loss_kd = MultiClassCrossEntropy(outputs, kd_outputs, T=2)
                loss = loss_ewc + self.kd_lambda * loss_kd

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected for client {self.cid}, task {actual_task_id}")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += local_y.size(0)
                    epoch_loss += loss.item()

        if actual_task_id > 0:
            fisher_old = {}
            if self.fisher is not None:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher:
                        fisher_old[n] = self.fisher[n].clone()
            self.fisher = self.fisher_matrix_diag(seq_idx, actual_task_id, loader)
            if self.fisher is not None and actual_task_id > 0 and len(fisher_old) > 0:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher and n in fisher_old:
                        self.fisher[n] = (self.fisher[n] + fisher_old[n] * actual_task_id) / (actual_task_id + 1)

        self.first_train = False
        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def pgd_train_ewc_kd_task(self, seq_idx, global_weights):
        """EWC+KD 本地训练 + PGD 对抗训练 (Elastic Weight Consolidation + Knowledge Distillation + PGD) - 使用全局标签映射"""
        actual_task_id = self.task_sequence[seq_idx]
        self.set_weights(global_weights)

        loader, _ = self.data_generator.get_loader(self.cid, actual_task_id, 'train')

        if actual_task_id != self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            for param in self.model_old.parameters():
                param.requires_grad = False
            self.old_task = actual_task_id
            self.cur_kd = deepcopy(self.kd_model)
            self.first_train = True

        self.train_kd(actual_task_id, loader)
        self.cur_kd.eval()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()
        epoch_loss, total_correct, total_samples = 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

        per_task_class_num = self.args.num_classes // self.args.num_tasks

        for epoch in range(self.args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = y - actual_task_id * per_task_class_num

                x_adv = self._get_adversarial_x(x, local_y, seq_idx,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)
                kd_outputs = self.cur_kd(x_adv, seq_idx)

                loss_ewc = self.ewc_criterion(actual_task_id, outputs, local_y)
                loss_kd = MultiClassCrossEntropy(outputs, kd_outputs, T=2)
                loss = loss_ewc + self.kd_lambda * loss_kd

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected for client {self.cid}, task {actual_task_id}")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == local_y).sum().item()
                    total_samples += local_y.size(0)
                    epoch_loss += loss.item()

        if actual_task_id > 0:
            fisher_old = {}
            if self.fisher is not None:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher:
                        fisher_old[n] = self.fisher[n].clone()
            self.fisher = self.fisher_matrix_diag(seq_idx, actual_task_id, loader)
            if self.fisher is not None and actual_task_id > 0 and len(fisher_old) > 0:
                for n, _ in self.model.named_parameters():
                    if n in self.fisher and n in fisher_old:
                        self.fisher[n] = (self.fisher[n] + fisher_old[n] * actual_task_id) / (actual_task_id + 1)

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        self.first_train = False
        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end

    def prune_kd_model(self, dataloader, task):
        """剪枝KD模型用于服务器聚合"""
        prune_state = {}
        for name, param in self.cur_kd.named_parameters():
            prune_state[name] = param.data.clone()
        # 保存 buffers（如 mu 和 std）
        for name, buf in self.cur_kd.named_buffers():
            prune_state[name] = buf.data.clone()
        return {'client': self.cid, 'model': prune_state, 'task': task}

    def compute_model_similarity(self, other_model_state, dataloader, task):
        """计算与另一个模型的相似度 (MultiClassCrossEntropy) - 使用全局标签"""
        temp_model = deepcopy(self.model)
        temp_model.load_state_dict(other_model_state)
        temp_model.eval()
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.args.device)
                out1 = self.model(images, task)
                out2 = temp_model(images, task)
                total_loss += MultiClassCrossEntropy(out1, out2)

        del temp_model
        return total_loss.item()

    def extensive_test(self, curr_round):
        """全面测试：评估 Clean 和 Robust 准确率 - 使用全局标签映射"""
        max_seq_idx = min(curr_round // self.args.t_round, self.args.num_tasks - 1)
        results = {'clean': [], 'robust': []}

        per_task_class_num = self.args.num_classes // self.args.num_tasks

        self.model.eval()
        for s_idx in range(max_seq_idx + 1):
            actual_tid = self.task_sequence[s_idx]
            loader, _ = self.data_generator.get_loader(self.cid, actual_tid, 'test')

            correct_clean, correct_adv, total = 0, 0, 0

            for x, y in loader:
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = y - actual_tid * per_task_class_num

                with torch.no_grad():
                    out_clean = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == local_y).sum().item()

                x_adv = self._get_adversarial_x(x, local_y, s_idx, steps=self.args.pgd_test)

                with torch.no_grad():
                    out_adv = self.model(x_adv, s_idx)
                    correct_adv += (out_adv.argmax(1) == local_y).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            results['robust'].append(correct_adv / total if total > 0 else 0)

        return results