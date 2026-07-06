import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import psutil

# --- 全局常量 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPSILON = 10.0 / 255.0  # UAP 扰动幅度限制


class Args:
    def __init__(self):
        # 系统设置
        self.device = DEVICE
        self.seed = 42
        self.state_dir = './states'
        self.data_root = '/data/dusiqi/datasets'

        # 联邦设置
        self.num_clients = 10
        self.num_rounds = 100  # 总通信轮次 CNN50
        self.t_round = 10 # cnn20 
        self.num_tasks =10 # 任务总数
        self.dirichlet_alpha = 0.5 # 0.5, 0.1, 0.01 越小异构越强
        self.num_workers = 0  # 建议设为0避免内存问题
        self.model='cnn' # cnn\vit\resnet\mobilenet

        # 数据集
        self.max_train_samples_per_client = 20000
        self.max_test_samples_per_client = 4000
        self.num_classes=100
        self.dataset='cifar100' # cifar10\cifar100\mnist

        # 训练超参
        self.batch_size = 64
        self.lr = 0.001
        self.local_epochs = 3  # 主任务训练轮数
        self.uap_gen_epochs = 2  # UAP 生成轮数
        self.adv_epochs = 2  # 对抗训练总轮数
        self.m_epochs = 1 # 对抗训练一阶段轮数
        self.adv_mix_ratio = 0.1  # 对抗样本混合比例 (20%)
        self.CONF_THRESHOLD = 0.1
        

        # 攻击
        self.attack='pgd'
        self.fgsm_eps=0.031 # 8/255
        self.pgd_eps=0.031
        self.pgd_alpha=0.007 # 2/255
        self.pgd_steps=10 # 7,10,20
        self.pgd_test=20 # 20, 100

        # FedWeIT 超参
        self.wd = 1e-4
        self.lambda_l1 = 1e-3
        self.lambda_mask = 1e-4
        self.lambda_l2 = 100.0
        self.sparsity_threshold = 1e-3  # 稀疏化阈值
        self.kb_sample_size = 3  # 知识库随机采样数量

        # LoRA & UAP 设置
        self.lora_rank = 4
        self.k_uap = 2
        self.uap_sim_threshold = 0.5  # 相似度合并阈值
        self.data_ratio = 0.1

        # DBFAT 超参
        self.dbfat_beta = 0.1  # KL散度损失权重
        self.dbfat_max_steps = 20  # PGD最大迭代步数（用于计算最小误分类步数）

        # 记录与输出设置
        self.record_comm_cost = True       # 通信开销计算模式开关
        self.target_robust_acc = 0.7      # 目标鲁棒精度（用于计算达到该精度的通信轮数）
        self.record_memory = True          # 内存占用记录开关


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_model_param_size(params):
    """计算参数总量（字节数）
    params: 可以是 model.state_dict() 或 parameters() 返回的参数字典/迭代器
    支持部分参数上传场景
    """
    total = 0
    if isinstance(params, torch.nn.Module):
        for param in params.parameters():
            total += param.nelement() * param.element_size()
    elif isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                total += value.nelement() * value.element_size()
    else:
        for param in params:
            if isinstance(param, torch.Tensor):
                total += param.nelement() * param.element_size()
    return total


def get_memory_usage():
    """获取当前进程内存占用（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class LogPrinter:
    """日志打印工具类，同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_print(self, *args, **kwargs):
        """同时打印到控制台和日志文件"""
        print(*args, **kwargs)
        with open(self.log_file, 'a') as f:
            print(*args, **kwargs, file=f)
    
    def __call__(self, *args, **kwargs):
        self.log_print(*args, **kwargs)


class Recorder:
    """统一的记录与输出类"""
    def __init__(self, log_print, args, comm_cost_per_round=0):
        self.log_print = log_print
        self.args = args
        self.comm_cost_per_round = comm_cost_per_round
        # 记录数据
        self.task_accuracy_history = {'clean': {}, 'robust': {}}
        self.task_comm_rounds = {}
        self.forgetting_rates = []
        self.time_stats = {'client_train': [], 'server_agg': []}
        self.memory_stats = {
            'client_train_peak': [],
            'client_train_end': [],
            'server_agg_peak': [],
            'server_agg_end': []
        }
        self.first_task_steady_mem = None
    
    def output_round_start(self, rnd):
        """输出轮次开始"""
        self.log_print(f"\n[Round {rnd + 1}/{self.args.num_rounds}]")
    
    def update_time_stats(self, client_train_time, server_agg_time):
        """更新时间统计"""
        self.time_stats['client_train'].append(client_train_time)
        self.time_stats['server_agg'].append(server_agg_time)
    
    def update_memory_stats(self, client_train_peak, client_train_end, server_agg_peak, server_agg_end):
        """更新内存统计"""
        self.memory_stats['client_train_peak'].append(client_train_peak)
        self.memory_stats['client_train_end'].append(client_train_end)
        self.memory_stats['server_agg_peak'].append(server_agg_peak)
        self.memory_stats['server_agg_end'].append(server_agg_end)
    
    def record_task_accuracy(self, global_task_id, clean_accs, robust_accs):
        """记录任务准确率"""
        for tid in range(global_task_id + 1):
            if tid not in self.task_accuracy_history['clean']:
                self.task_accuracy_history['clean'][tid] = []
                self.task_accuracy_history['robust'][tid] = []
            self.task_accuracy_history['clean'][tid].append(clean_accs[tid])
            self.task_accuracy_history['robust'][tid].append(robust_accs[tid])
    
    def check_comm_cost_target(self, global_task_id, robust_acc, current_round):
        """检查是否达到通信开销目标"""
        if self.args.record_comm_cost and global_task_id not in self.task_comm_rounds:
            if robust_acc >= self.args.target_robust_acc:
                self.task_comm_rounds[global_task_id] = current_round + 1
    
    def output_task_switch(self, prev_task_id, curr_task_id):
        """输出任务切换信息"""
        self.log_print(f"\n  >> Task {prev_task_id} completed, entering Task {curr_task_id}")
        
        global_forgetting_rates = []
        self.log_print(f"  History Tasks Forgetting Rates:")
        for task_id in range(curr_task_id):
            if task_id in self.task_accuracy_history['robust']:
                acc_history = self.task_accuracy_history['robust'][task_id]
                max_acc = max(acc_history)
                current_acc = acc_history[-1]
                task_forgetting_rate = max_acc - current_acc
                global_forgetting_rates.append(task_forgetting_rate)
                self.log_print(f"    Task {task_id}: max_acc={max_acc:.4f}, current_acc={current_acc:.4f}, forgetting_rate={task_forgetting_rate:.4f}")
        
        if global_forgetting_rates:
            self.log_print(f"  Global Avg Forgetting Rate: {np.mean(global_forgetting_rates):.4f}")
            self.log_print(f"  Global Total Forgetting Rate: {np.sum(global_forgetting_rates):.4f}")
        
        if self.args.record_comm_cost:
            if prev_task_id not in self.task_comm_rounds:
                rounds_trained = (curr_task_id - prev_task_id) * self.args.t_round
                self.task_comm_rounds[prev_task_id] = rounds_trained
            comm_cost = self.task_comm_rounds[prev_task_id] * self.comm_cost_per_round
            self.log_print(f"  Task {prev_task_id} Communication Cost: {self.task_comm_rounds[prev_task_id]} rounds × {self.comm_cost_per_round/1024/1024:.2f} MB = {comm_cost/1024/1024:.2f} MB")
        
        current_task_forgetting_rates = []
        for task_id in range(prev_task_id + 1):
            if task_id in self.task_accuracy_history['robust']:
                acc_history = self.task_accuracy_history['robust'][task_id]
                max_acc = max(acc_history)
                current_acc = acc_history[-1]
                task_forgetting_rate = max_acc - current_acc
                current_task_forgetting_rates.append(task_forgetting_rate)
        
        if current_task_forgetting_rates:
            self.forgetting_rates.append(np.mean(current_task_forgetting_rates))
        
        if self.args.record_memory and self.first_task_steady_mem is None:
            self.first_task_steady_mem = np.mean(self.memory_stats['client_train_end'])
    
    def output_client_train(self, client_cid, client_task_id, loss, train_acc, train_time):
        """输出客户端训练信息"""
        self.log_print(f"    Client {client_cid}: Task {client_task_id}, Loss={loss:.4f}, Train Acc={train_acc:.2f}, Time={train_time:.2f}s")
    
    def output_server_agg(self, agg_time):
        """输出服务器聚合信息"""
        self.log_print("  > Step 2: Global Aggregation")
        self.log_print(f"    Aggregation Time: {agg_time:.4f}s")
    
    def output_client_eval(self, client_cid, res):
        """输出客户端评估结果"""
        self.log_print(f"    Client {client_cid} Results (Global Tasks 0-{len(res['clean']) - 1}):")
        self.log_print(f"      Clean Acc    : {[f'{a:.2f}' for a in res['clean']]} (Avg: {np.mean(res['clean']):.2f})")
        self.log_print(f"      PGD-Adv Acc  : {[f'{a:.2f}' for a in res['robust']]} (Avg: {np.mean(res['robust']):.2f})")
    
    def output_round_summary(self, rnd, all_results):
        """输出轮次总结"""
        self.log_print(f"\n  >> Round {rnd + 1} Global Summary:")
        self.log_print(f"     Mean Clean: {np.mean(all_results['clean']):.4f}")
        self.log_print(f"     Mean PGD-Adv: {np.mean(all_results['robust']):.4f}")
        self.log_print(f"     Time - Client Train: {self.time_stats['client_train'][-1]:.2f}s, Server Agg: {self.time_stats['server_agg'][-1]:.4f}s")
        if self.args.record_memory:
            self.log_print(f"     Memory - Client Peak: {self.memory_stats['client_train_peak'][-1]:.2f}MB, Steady: {self.memory_stats['client_train_end'][-1]:.2f}MB")
            self.log_print(f"              Server Peak: {self.memory_stats['server_agg_peak'][-1]:.2f}MB, Steady: {self.memory_stats['server_agg_end'][-1]:.2f}MB")
    
    def output_summary(self, round_acc_clean, round_acc_robust, global_task_id, total_time, output_dir):
        """输出完整总结"""
        self.log_print("\n" + "="*60)
        self.log_print("[Round Summary]")
        self.log_print("="*60)
        self.log_print(f"Clean Acc per Round    : {[f'{a:.2f}' for a in round_acc_clean]}")
        self.log_print(f"PGD-Adv Acc per Round  : {[f'{a:.2f}' for a in round_acc_robust]}")
        
        if self.forgetting_rates:
            self.log_print(f"\n[Forgetting Rate Summary]")
            self.log_print(f"  Forgetting Rates per Task Switch: {[f'{fr:.4f}' for fr in self.forgetting_rates]}")
            self.log_print(f"  Final Forgetting Rate: {self.forgetting_rates[-1]:.4f}")
            self.log_print(f"  Overall Average Forgetting Rate: {np.mean(self.forgetting_rates):.4f}")
        
        if self.args.record_comm_cost and global_task_id not in self.task_comm_rounds:
            self.task_comm_rounds[global_task_id] = self.args.num_rounds - global_task_id * self.args.t_round
        
        if self.args.record_comm_cost:
            total_comm_cost = 0
            self.log_print(f"\n[Communication Cost Summary]")
            self.log_print(f"  Per Round Cost: {self.comm_cost_per_round / 1024 / 1024:.2f} MB (upload + download)")
            for tid in sorted(self.task_comm_rounds.keys()):
                rounds = self.task_comm_rounds[tid]
                cost = rounds * self.comm_cost_per_round
                total_comm_cost += cost
                self.log_print(f"  Task {tid}: {rounds} rounds × {self.comm_cost_per_round / 1024 / 1024:.2f} MB = {cost / 1024 / 1024:.2f} MB")
            self.log_print(f"  Total Communication Cost: {total_comm_cost / 1024 / 1024:.2f} MB")
        
        self.log_print(f"\n[Timer Summary]")
        self.log_print(f"  Total Time: {total_time / 60:.2f} min")
        self.log_print(f"  Client Training Time: {np.sum(self.time_stats['client_train']):.2f}s")
        self.log_print(f"  Server Aggregation Time: {np.sum(self.time_stats['server_agg']):.2f}s")
        
        if self.args.record_memory:
            last_task_steady_mem = np.mean(self.memory_stats['client_train_end'])
            steady_mem_growth = last_task_steady_mem - self.first_task_steady_mem if self.first_task_steady_mem else 0
            self.log_print(f"\n[Memory Summary]")
            self.log_print(f"  Client Train - Peak: {np.max(self.memory_stats['client_train_peak']):.2f}MB")
            self.log_print(f"  Steady Memory Growth (Task 1 -> Task {self.args.num_tasks}): {steady_mem_growth:.2f}MB")
            self.log_print(f"  Server Agg - Peak: {np.max(self.memory_stats['server_agg_peak']):.2f}MB")
        
        self.log_print(f"\n" + "="*60)
        self.log_print(f"Results saved to {output_dir}")
        self.log_print("="*60)


def compute_cosine_similarity(uap1, uap2):
    """ 计算两个 UAP 的余弦相似度 """
    v1 = uap1.view(-1)
    v2 = uap2.view(-1)
    if torch.norm(v1) == 0 or torch.norm(v2) == 0: return 0.0
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

def set_lora_active(model, active=True):
    """ 批量设置模型中所有层的 LoRA 开关状态 """
    for module in model.modules():
        if hasattr(module, 'use_lora'):
            module.use_lora = active

class UAPGenerator:
    """ 通用对抗扰动生成器 """

    def __init__(self, model, task_id):
        self.model = model
        self.task_id = task_id
        # 初始化扰动 (Requires Grad)
        self.delta = torch.zeros((1, 3, 32, 32), device=DEVICE, requires_grad=True)
        self.raw_delta = torch.zeros(1, 3, 32, 32)

    def run(self, loader, label_map=None, epochs=1, data_ratio=1.0):  # <--- 新增 label_map 参数
        self.model.eval()
        opt = optim.Adam([self.delta], lr=0.05)

        # --- 新增：按比例选择数据 ---
        if data_ratio < 1.0:
            total_len = len(loader.dataset)
            subset_len = int(total_len * data_ratio)
            indices = torch.randperm(total_len)[:subset_len]
            from torch.utils.data import Subset
            subset = Subset(loader.dataset, indices)
            uap_loader = torch.utils.data.DataLoader(
                subset, batch_size=loader.batch_size, shuffle=True,
                num_workers=loader.num_workers, pin_memory=loader.pin_memory
            )
            print(f"Using {subset_len}/{total_len} images ({data_ratio * 100:.0f}%)")
        else:
            uap_loader = loader

        for _ in range(epochs):
            for img, label in uap_loader:
                img = img.to(DEVICE)

                # --- [核心修复] ---
                if label_map is not None:
                    # 将全局 label 映射为局部 label
                    # 注意：要处理 Tensor 在 GPU 上的转换
                    label = torch.tensor([label_map[l.item()] for l in label], device=DEVICE)
                else:
                    label = label.to(DEVICE)
                # -----------------

                # 叠加扰动
                adv_imgs = torch.clamp(img + self.delta, 0, 1)

                logits = self.model(adv_imgs, self.task_id)
                loss = -F.cross_entropy(logits, label)

                opt.zero_grad()
                loss.backward()
                opt.step()
                # --- 记录约束前的原始更新结果 ---
                self.raw_delta = self.delta.detach().clone()  # ! 捕获 Adam 更新后但未 Clamp 的状态

                # 执行约束
                self.delta.data.clamp_(-EPSILON, EPSILON)

                # 返回最终扰动和最后一个 batch 的原始更新向量
        return self.raw_delta