"""
本地机器学习训练与对抗攻击评估脚本

修改说明：
1. 移除联邦学习（FCL）逻辑，改为单客户端本地训练
2. 简化数据加载方式，直接从数据集随机抽取指定数量样本
3. 添加对抗攻击评估模块，计算不同攻击的计算开销和欺骗率

实验设置：
- 数据集：CIFAR-10
- 预训练：2000张图片训练20轮，批大小64，学习率0.001
- 提前停止：准确率达到0.99时自动停止
- 对抗攻击测试：64张图片，评估5种攻击方法

评估指标：
- 计算开销：生成64张对抗样本所耗费的时间（秒）
- 欺骗率：1 - 对抗样本上的模型准确率

支持的攻击方法：
- PGD (Projected Gradient Descent)
- FGSM (Fast Gradient Sign Method)
- JSMA (Jacobian-based Saliency Map Attack)
- DeepFool
- AutoAttack (集成攻击)

输出：
- 控制台实时日志
- 日志文件：outputs/{model}_{dataset}/{date}/local_train_*.log
- 汇总表格：包含各攻击的计算开销和欺骗率
"""

from utils import Args, set_seed, LogPrinter
from nets import Init_model
from data_loader import TaskGenerator
from attack import Attack
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import time


def main():
    args = Args()
    set_seed(args.seed)

    now = datetime.now()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"outputs/{args.model}_{args.dataset}/{now.strftime('%Y-%m-%d')}")
    os.makedirs(output_dir, exist_ok=True)
    
    params_str = f"local_train_{args.dataset}_samples{args.train_samples}_epochs{args.epochs}_batch{args.batch_size}_lr{args.lr}_{now.strftime('%H%M%S')}"
    log_file = os.path.join(output_dir, f'{params_str}_log.txt')
    
    log_print = LogPrinter(log_file)

    log_print(f"Output directory: {output_dir}")
    log_print(f"Log file: {log_file}")
    log_print("Initializing Local Training with Adversarial Attack Evaluation...")

    task_gen = TaskGenerator(args)
    model = Init_model(args).to(args.device)

    log_print(f"\n=== Local Training on {args.dataset.upper()} ===")
    log_print(f"Training samples: {args.train_samples}")
    log_print(f"Test samples for attack: {args.test_samples}")
    log_print(f"Epochs: {args.epochs}")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Learning rate: {args.lr}")

    train_loader = task_gen.get_loader(mode='train', num_samples=args.train_samples)
    test_loader = task_gen.get_loader(mode='test', num_samples=args.test_samples)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    log_print("\n--- Pre-training Phase ---")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
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
        
        epoch_time = time.time() - start_time
        epoch_loss /= total_samples
        epoch_acc = total_correct / total_samples
        
        log_print(f"Epoch {epoch+1}/{args.epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Time={epoch_time:.2f}s")

        if epoch_acc >= 0.99:
            log_print(f"Early stopping at epoch {epoch+1} with accuracy {epoch_acc:.4f}")
            break

    model.eval()
    log_print("\n--- Final Clean Accuracy on Training Data ---")
    train_correct = 0
    train_total = 0
    for x, y in train_loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            outputs = model(x)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)
    train_acc = train_correct / train_total
    log_print(f"Training Accuracy: {train_acc:.4f}")

    log_print("\n--- Adversarial Attack Evaluation ---")
    log_print(f"Testing on {args.test_samples} images")

    all_attacks = ['fgsm', 'pgd', 'mi_fgsm', 'df_uap', 'deepfool', 'cw']
    
    attack_results = {}
    for attack_name in all_attacks:
        log_print(f"\nEvaluating {attack_name.upper()} attack...")
        
        if attack_name == 'df_uap':
            attacker = Attack(model, args, proxy_loader=train_loader)
        else:
            attacker = Attack(model, args)
        model.eval()
        
        correct_adv = 0
        total = 0
        attack_start_time = time.time()
        
        for x, y in test_loader:
            if x.size(0) == 1:
                continue
            x, y = x.to(args.device), y.to(args.device)
            
            x_adv = attacker.attack(x, y, attack_name=attack_name)
            
            with torch.no_grad():
                outputs = model(x_adv)
                correct_adv += (outputs.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        
        attack_time = time.time() - attack_start_time
        robust_acc = correct_adv / total if total > 0 else 0
        fooling_rate = 1 - robust_acc
        
        attack_results[attack_name] = {
            'robust_acc': robust_acc,
            'fooling_rate': fooling_rate,
            'time': attack_time
        }
        
        log_print(f"  {attack_name.upper()} Attack Time: {attack_time:.4f}s")
        log_print(f"  {attack_name.upper()} Robust Accuracy: {robust_acc:.4f}")
        log_print(f"  {attack_name.upper()} Fooling Rate: {fooling_rate:.4f}")

    log_print("\n--- Clean Accuracy on Test Data ---")
    clean_correct = 0
    clean_total = 0
    for x, y in test_loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            outputs = model(x)
            clean_correct += (outputs.argmax(dim=1) == y).sum().item()
        clean_total += x.size(0)
    clean_acc = clean_correct / clean_total
    log_print(f"Clean Accuracy: {clean_acc:.4f}")

    log_print("\n" + "="*60)
    log_print("[Attack Evaluation Summary]")
    log_print("="*60)
    log_print(f"Dataset: {args.dataset.upper()}")
    log_print(f"Training samples: {args.train_samples}")
    log_print(f"Test samples: {args.test_samples}")
    log_print(f"Pre-training epochs: {args.epochs}")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Learning rate: {args.lr}")
    log_print(f"\nClean Accuracy on Test Set: {clean_acc:.4f}")
    log_print("\nAttack Results:")
    log_print(f"{'Attack':<15} {'Time(s)':<10} {'Robust Acc':<12} {'Fooling Rate':<15}")
    log_print("-" * 52)
    for attack_name in all_attacks:
        result = attack_results[attack_name]
        log_print(f"{attack_name.upper():<15} {result['time']:<10.4f} {result['robust_acc']:<12.4f} {result['fooling_rate']:<15.4f}")
    log_print("="*60)


if __name__ == '__main__':
    main()