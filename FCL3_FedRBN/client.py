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
        self.is_AT = False
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
        if global_weights is not None:
            self.model.load_state_dict(global_weights)

    def set_AT_weights(self, non_bn_weights):
        if non_bn_weights is not None:
            self.model.load_state_dict(non_bn_weights, strict=False)

    def set_ST_weights(self, non_bn_weights, bn_a_update):
        if non_bn_weights is not None:
            self.model.load_state_dict(non_bn_weights, strict=False)
        if bn_a_update is not None:
            for key in bn_a_update:
                if key in self.model.state_dict():
                    self.model.state_dict()[key].copy_(bn_a_update[key])

    def _separate_bn_params(self, state_dict):
        bn_params = {}
        non_bn_params = {}
        for key, value in state_dict.items():
            if 'bn' in key.lower() or 'batch_norm' in key.lower():
                bn_params[key] = value
            else:
                non_bn_params[key] = value
        return non_bn_params, bn_params

    def _get_bn_c_mean_var(self):
        features = self.model.base_model.features if hasattr(self.model, 'base_model') else self.model.features
        return features.get_bn_c_mean_var() if hasattr(features, 'get_bn_c_mean_var') else {}

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



    def AT_train_main_task(self, seq_idx):
        actual_task_id = self.task_sequence[seq_idx]
        self.is_AT = True

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}

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
                y_map = self._map_global_to_local_label(y, actual_task_id)


                x_adv = self._get_adversarial_x(x, y_map, actual_task_id, steps=self.args.pgd_steps)

                optimizer.zero_grad()

                outputs_c = self.model(x, seq_idx, use_bn_a=False)
                outputs_a = self.model(x_adv, seq_idx, use_bn_a=True)

                loss_c = criterion(outputs_c, y_map)
                loss_a = criterion(outputs_a, y_map)
                loss = loss_c + loss_a

                loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (outputs_c.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        state_dict = self.model.state_dict()
        non_bn_params, bn_params_dict = self._separate_bn_params(state_dict)
        bn_c_mean_var = self._get_bn_c_mean_var()

        return non_bn_params, bn_params_dict, bn_c_mean_var, epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end

    def ST_train_main_task(self, seq_idx):
        actual_task_id = self.task_sequence[seq_idx]
        self.is_AT = False

        loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'train')
        label_map = {g: l for l, g in enumerate(classes)}

        non_bn_params = []
        bn_c_params = []
        for name, param in self.model.named_parameters():
            if 'bn_a' in name:
                param.requires_grad = False
            elif 'bn_c' in name:
                bn_c_params.append(param)
                param.requires_grad = True
            else:
                non_bn_params.append(param)
                param.requires_grad = True

        optimizer = optim.Adam(non_bn_params + bn_c_params, lr=self.args.lr)
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
                y_map = self._map_global_to_local_label(y, actual_task_id)

                optimizer.zero_grad()

                outputs_c = self.model(x, seq_idx, use_bn_a=False)
                outputs_a = self.model(x, seq_idx, use_bn_a=True)

                loss_c = criterion(outputs_c, y_map)
                loss_a = criterion(outputs_a, y_map)
                lambda_pnc = getattr(self.args, 'lambda_pnc', 0.5)
                loss = (1 - lambda_pnc) * loss_c + lambda_pnc * loss_a

                loss.backward()
                optimizer.step()

                if self.args.record_memory:
                    current_mem = get_memory_usage()
                    if current_mem > mem_peak:
                        mem_peak = current_mem

                with torch.no_grad():
                    total_correct += (outputs_c.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        training_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        for name, param in self.model.named_parameters():
            if 'bn_a' in name:
                param.requires_grad = True

        state_dict = self.model.state_dict()
        non_bn_params_dict, bn_params_dict = self._separate_bn_params(state_dict)
        bn_c_mean_var = self._get_bn_c_mean_var()

        return non_bn_params_dict, bn_params_dict, bn_c_mean_var, epoch_loss / total_samples, total_correct / total_samples, training_time, mem_start, mem_peak, mem_end

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
                x_adv = self._get_adversarial_x(x, y_map, seq_idx,
                                                attack_name='pgd',
                                                steps=self.args.pgd_steps)

                optimizer.zero_grad()
                self.model.train()
                outputs = self.model(x_adv, seq_idx)
                loss = criterion(outputs, y_map)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_correct += (outputs.argmax(dim=1) == y_map).sum().item()
                    total_samples += y.size(0)
                    epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / total_samples, total_correct / total_samples

    def extensive_test(self, curr_round):
        """全面测试：评估 Clean 和 Robust 准确率"""
        max_seq_idx = min(curr_round // self.args.t_round, self.args.num_tasks - 1)
        results = {'clean': [], 'robust': []}

        self.model.eval()
        for s_idx in range(max_seq_idx + 1):
            actual_task_id = self.task_sequence[s_idx]
            loader, classes = self.data_generator.get_loader(self.cid, actual_task_id, 'test')
            label_map = {g: l for l, g in enumerate(classes)}

            correct_clean, correct_adv, total = 0, 0, 0

            for x, y in loader:
                if x.size(0)==1: # 注意：部分模型要求输入batch_size>1
                    continue
                x, y = x.to(self.args.device), y.to(self.args.device)
                y_map = self._map_global_to_local_label(y, actual_task_id)

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