import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from utils import get_memory_usage
from attack import Attack
import copy

class Client:
    def __init__(self, cid, args, model, data_loader, task_sequence):
        self.cid = cid
        self.args = args
        self.model = model.to(args.device)
        self.data_generator = data_loader
        self.task_sequence = task_sequence
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

    def _get_adversarial_x(self, x, y, task_id=None, attack_name=None, steps=None):
        """
        统一攻击入口：显式分支判断（仅保留 fgsm 和 pgd 用于对抗训练）
        task_id 参数保留但不强制使用
        """
        name = attack_name if attack_name else self.args.attack

        if name == 'pgd':
            return self.pgd(x, y, task_id, steps=steps)
        elif name == 'fgsm':
            return self.fgsm(x, y, task_id)
        else:
            raise ValueError(f"Unsupported attack for training: {name}")

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

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

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

    def pgd_train_main_task(self, seq_idx, global_weights):
        """对抗训练 (PGD Adversarial Training)
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

        mem_start = get_memory_usage() if self.args.record_memory else 0
        mem_peak = mem_start
        start_time = time.time()

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
        """全面测试：评估 Clean 和各种攻击下的鲁棒准确率
        s_idx: 全局任务ID（当前训练的第几个任务，所有客户端相同）
        actual_tid: 本地真实任务ID（数据集中对应的任务ID，各客户端不同）
        """
        all_attacks = ['pgd', 'fgsm', 'jsma', 'deepfool','autoattack']
        
        if curr_round == self.args.num_rounds - 1:
            attacks = all_attacks
        else:
            if self.args.attack in all_attacks:
                attacks = [self.args.attack]
            else:
                attacks = ['pgd']
        
        max_seq_idx = min(curr_round // self.args.t_round, self.args.num_tasks - 1)
        results = {'clean': []}
        for attack in attacks:
            results[attack] = []

        attacker = Attack(copy.deepcopy(self.model), self.args)

        self.model.eval()
        for s_idx in range(max_seq_idx + 1):
            actual_tid = self.task_sequence[s_idx]
            loader, classes = self.data_generator.get_loader(self.cid, actual_tid, 'test')

            correct_clean = 0
            correct_adv = {attack: 0 for attack in attacks}
            total = 0

            for x, y in loader:
                if x.size(0) == 1:
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)

                local_y = self._map_global_to_local_label(y, actual_tid)

                with torch.no_grad():
                    out_clean = self.model(x, s_idx)
                    correct_clean += (out_clean.argmax(1) == local_y).sum().item()

                for attack in attacks:
                    x_adv = attacker.attack(x, local_y, s_idx, attack_name=attack)
                    with torch.no_grad():
                        out_adv = self.model(x_adv, s_idx)
                        correct_adv[attack] += (out_adv.argmax(1) == local_y).sum().item()

                total += x.size(0)

            results['clean'].append(correct_clean / total if total > 0 else 0)
            for attack in attacks:
                results[attack].append(correct_adv[attack] / total if total > 0 else 0)

        return results