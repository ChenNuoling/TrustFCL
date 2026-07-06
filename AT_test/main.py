"""
对抗训练方法评估脚本

实验设置：
- 数据集：CIFAR-10
- 训练样本：2000张图片，批大小64，学习率0.001
- 对抗攻击测试：64张图片，PGD-20攻击，ε=8/255
- 评估指标：计算开销（训练时间）、鲁棒精度提升

支持的对抗训练方法（9种）：
- AT: Adversarial Training (Madry et al., 2018)
- TRADES: Trust Region-based Adversarial Training (Zhang et al., 2019)
- MART: Improving Adversarial Robustness (Wang et al., 2020)
- GAIRAT: Gradient Aligned Intermediate Representation AT (Li et al., 2020)
- FAT: Feature-wise Adversarial Training (Cui et al., 2019)
- Free-AT: Free Adversarial Training (Shafahi et al., 2019)
- YOPO: You Only Propagate Once (Cheng et al., 2020)
- Shared AT: Shared Adversarial Training (Zhang et al., 2021)
- UIAT: Universal Instance-wise Adversarial Training (Wang et al., 2021)

输出：
- 控制台实时日志
- 日志文件：outputs/{model}_{dataset}/{date}/adv_train_eval_*.log
- 汇总表格：包含各方法的计算开销和鲁棒精度提升
"""

from utils import Args, set_seed, LogPrinter
from nets import Init_model
from data_loader import TaskGenerator
from attack import Attack
from ATrain import AdvTrain
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import time


def train_baseline(model, train_loader, args):
    """训练普通基线模型"""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for x, y in train_loader:
            if x.size(0) == 1:
                continue
            x, y = x.to(args.device), y.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            total_correct += (outputs.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
        
        epoch_loss /= total_samples
        epoch_acc = total_correct / total_samples
        
        if epoch_acc >= 0.99:
            break
    
    train_time = time.time() - start_time
    return train_time


def evaluate_robustness(model, test_loader, args):
    """评估模型在PGD-20攻击下的鲁棒精度"""
    model.eval()
    attacker = Attack(model, args)
    
    correct_adv = 0
    total = 0
    
    for x, y in test_loader:
        if x.size(0) == 1:
            continue
        x, y = x.to(args.device), y.to(args.device)
        
        x_adv = attacker.pgd(x, y, steps=20)
        
        with torch.no_grad():
            outputs = model(x_adv)
            correct_adv += (outputs.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    
    robust_acc = correct_adv / total if total > 0 else 0
    return robust_acc


def evaluate_clean_acc(model, test_loader, args):
    """评估模型在干净样本上的精度"""
    model.eval()
    correct = 0
    total = 0
    
    for x, y in test_loader:
        if x.size(0) == 1:
            continue
        x, y = x.to(args.device), y.to(args.device)
        
        with torch.no_grad():
            outputs = model(x)
            correct += (outputs.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    
    clean_acc = correct / total if total > 0 else 0
    return clean_acc


def main():
    args = Args()
    set_seed(args.seed)

    now = datetime.now()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"outputs/{args.model}_{args.dataset}/{now.strftime('%Y-%m-%d')}")
    os.makedirs(output_dir, exist_ok=True)
    
    params_str = f"adv_train_eval_{args.dataset}_samples{args.train_samples}_epochs{args.epochs}_batch{args.batch_size}_lr{args.lr}_{now.strftime('%H%M%S')}"
    log_file = os.path.join(output_dir, f'{params_str}_log.txt')
    
    log_print = LogPrinter(log_file)

    log_print(f"Output directory: {output_dir}")
    log_print(f"Log file: {log_file}")
    log_print("Initializing Adversarial Training Evaluation...")

    task_gen = TaskGenerator(args)

    log_print(f"\n=== Experiment Settings ===")
    log_print(f"Dataset: {args.dataset.upper()}")
    log_print(f"Training samples: {args.train_samples}")
    log_print(f"Test samples for attack: {args.test_samples}")
    log_print(f"Epochs: {args.epochs}")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Learning rate: {args.lr}")
    log_print(f"Attack: PGD-20, ε=8/255")

    train_loader = task_gen.get_loader(mode='train', num_samples=args.train_samples)
    test_loader = task_gen.get_loader(mode='test', num_samples=args.test_samples)

    at_methods = [
        ('AT', 'at'),
        ('TRADES', 'trades'),
        ('MART', 'mart'),
        ('GAIRAT', 'gairat'),
        ('FAT', 'fat'),
        ('Free-AT', 'free_at'),
        ('YOPO', 'yopo'),
        ('Shared AT', 'shared_at'),
        ('UIAT', 'uiat')
    ]

    results = {}

    log_print("\n--- Step 1: Training Baseline Model ---")
    baseline_model = Init_model(args).to(args.device)
    baseline_time = train_baseline(baseline_model, train_loader, args)
    baseline_clean_acc = evaluate_clean_acc(baseline_model, test_loader, args)
    baseline_robust_acc = evaluate_robustness(baseline_model, test_loader, args)
    
    log_print(f"Baseline Training Time: {baseline_time:.4f}s")
    log_print(f"Baseline Clean Accuracy: {baseline_clean_acc:.4f}")
    log_print(f"Baseline Robust Accuracy (PGD-20): {baseline_robust_acc:.4f}")
    
    results['Baseline'] = {
        'train_time': baseline_time,
        'clean_acc': baseline_clean_acc,
        'robust_acc': baseline_robust_acc,
        'robust_improvement': 0.0
    }

    log_print("\n--- Step 2: Evaluating Adversarial Training Methods ---")
    for method_name, method_key in at_methods:
        log_print(f"\n{'='*60}")
        log_print(f"Training with {method_name}...")
        log_print(f"{'='*60}")
        
        model = Init_model(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        adv_trainer = AdvTrain(model, args)
        train_time = adv_trainer.train(train_loader, optimizer, args.epochs, method=method_key)
        
        clean_acc = evaluate_clean_acc(model, test_loader, args)
        robust_acc = evaluate_robustness(model, test_loader, args)
        robust_improvement = robust_acc - baseline_robust_acc
        
        log_print(f"{method_name} Training Time: {train_time:.4f}s")
        log_print(f"{method_name} Clean Accuracy: {clean_acc:.4f}")
        log_print(f"{method_name} Robust Accuracy (PGD-20): {robust_acc:.4f}")
        log_print(f"{method_name} Robust Improvement: {robust_improvement:.4f}")
        
        results[method_name] = {
            'train_time': train_time,
            'clean_acc': clean_acc,
            'robust_acc': robust_acc,
            'robust_improvement': robust_improvement
        }
        
        del model
        del optimizer
        del adv_trainer
        torch.cuda.empty_cache()

    log_print("\n" + "="*80)
    log_print("[Adversarial Training Evaluation Summary]")
    log_print("="*80)
    log_print(f"Dataset: {args.dataset.upper()}")
    log_print(f"Training samples: {args.train_samples}")
    log_print(f"Test samples: {args.test_samples}")
    log_print(f"Epochs: {args.epochs}")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Learning rate: {args.lr}")
    log_print(f"Attack: PGD-20, ε=8/255")
    log_print()
    log_print(f"Baseline Clean Accuracy: {baseline_clean_acc:.4f}")
    log_print(f"Baseline Robust Accuracy: {baseline_robust_acc:.4f}")
    log_print()
    log_print(f"{'Method':<15} {'Train Time(s)':<15} {'Clean Acc':<12} {'Robust Acc':<12} {'Improvement':<15}")
    log_print("-" * 70)
    log_print(f"{'Baseline':<15} {baseline_time:<15.4f} {baseline_clean_acc:<12.4f} {baseline_robust_acc:<12.4f} {0.0:<15.4f}")
    
    for method_name, _ in at_methods:
        result = results[method_name]
        log_print(f"{method_name:<15} {result['train_time']:<15.4f} {result['clean_acc']:<12.4f} {result['robust_acc']:<12.4f} {result['robust_improvement']:<15.4f}")
    
    log_print("="*80)


if __name__ == '__main__':
    main()