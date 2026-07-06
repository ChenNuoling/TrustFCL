from utils import Args, set_seed, get_model_param_size, LogPrinter, Recorder
from nets import Init_model
from data_loader import TaskGenerator
from client import Client
from server import Server
import numpy as np
import copy
import time
import torch
import os
import json
from datetime import datetime

"""
Sylva方法总结：
1、采用LoRA进行本地对抗训练，仅更新LoRA和分类器，
损失函数包括：
1）加权交叉熵损失，对少数类样本赋予更高权重
2）KL 散度损失（保持对干净样本与对抗样本输出的一致性）
3） L2 正则化（限制 LoRA 参数不要偏离全局模型太远）；
2、客户端仅上传 LoRA 参数至服务器，服务器将 LoRA 参数向量化，构建球树（一种空间索引结构，
将高维向量递归地划分为超球体，每个节点代表一个子集，以便
高效地查找与目标向量最相似的k个向量），将最相似的k 个客户端计算距离，
使用高斯加权聚合；
3、为了在不牺牲鲁棒性的前提下，提升干净准确率，在每一轮任务训练结束后（t_round轮次后），
利用 Shapley 值量化模型每一层对鲁棒性与准确性的边际贡献，选择一部分层进行训练，其余冻结。
计算方式：定义一组层的Shapley 值为干净损失价值（训练该组层后，
模型在干净样本上的交叉熵损失变化）-β*鲁棒性损失价值（训练该组层后，
模型在对抗样本上的交叉熵损失变化）。理论上需要枚举所有子集，
但通过蒙特卡洛采样近似，随机生成B个层排列，对每一层 计算其在排列中加入当前集合的边际贡献平均值，
重复B次（如 300 次），取平均，选出 Shapley 值最高的p个层（如 3% 的层），
冻结其余所有层，仅用干净样本训练这些选中的层。

更改的位置说明：（FCL\FCL_Sylva下的代码）
1、在net.py中，添加一个函数，该函数在模型初始化时调用（main.py初始化调用），用于添加LoRA参数。
2、在client.py中,将pgd_train_main_task替换为sylva_train_main_task,对抗训练改为
只训练LoRA和分类器，损失函数改为Sylva的三个，传回的参数改为LoRA参数。
3、在server.py中,将aggregate函数更改为符合Sylva的聚合函数，构建球树索引，并用于
找相似的LoRA向量，高斯加权聚合LoRA参数。
4、在main.py中每一次聚合后测试前，加一个判断，当前任务是否结束训练，如果是，添加一个二阶段训练，
每个client调用client.py中的函数Sylva_retrain()
5、在client.py中，添加一个函数Sylva_retrain(),该函数在二阶段训练中调用，用于
用Shapley值计算出需要训练的层，冻结其余层，仅用干净样本微调这些选中的层。
"""


def main():
    args = Args()
    set_seed(args.seed)

    now = datetime.now()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"outputs/{args.model}_{args.dataset}/{args.attack}/{now.strftime('%Y-%m-%d')}")
    os.makedirs(output_dir, exist_ok=True)
    
    params_str = f"clients{args.num_clients}_tround{args.t_round}_tasks{args.num_tasks}_alpha{args.dirichlet_alpha}_batch{args.batch_size}_lr{args.lr}_epochs{args.local_epochs}_{now.strftime('%H%M%S')}"
    log_file = os.path.join(output_dir, f'{params_str}_log.txt')
    
    log_print = LogPrinter(log_file)

    log_print(f"Output directory: {output_dir}")
    log_print(f"Log file: {log_file}")
    log_print("Initializing Sylva FCL components with Randomized Task Sequences...")
    task_gen = TaskGenerator(args)

    dummy_model = Init_model(args)
    server = Server(args)
    global_weights = copy.deepcopy(dummy_model.state_dict())

    all_task_ids = list(range(args.num_tasks))
    client_task_sequences = [np.random.permutation(all_task_ids).tolist() for _ in range(args.num_clients)]
    log_print(f"Client task sequences: {client_task_sequences}")
    clients = [
        Client(i, args, copy.deepcopy(dummy_model), task_gen, task_sequence=client_task_sequences[i])
        for i in range(args.num_clients)
    ]

    log_print("=== Federated Continual Learning with Sylva (Randomized Sequences) ===")

    total_start_time = time.time()
    round_acc_clean = []
    round_acc_robust = []

    prev_global_task_id = -1

    comm_cost_per_round = 0
    recorder = Recorder(log_print, args, comm_cost_per_round)

    for rnd in range(args.num_rounds):
        recorder.output_round_start(rnd)

        global_task_id = min(rnd // args.t_round, args.num_tasks - 1)

        if global_task_id != prev_global_task_id and prev_global_task_id >= 0:
            recorder.output_task_switch(prev_global_task_id, global_task_id)
        prev_global_task_id = global_task_id

        log_print("  > Step 1: Sylva Local Training")
        reports = []
        client_train_times = []
        client_mem_peaks = []
        client_mem_ends = []

        for client in clients:
            rep, loss, train_acc, train_time, mem_start, mem_peak, mem_end = client.sylva_train_main_task(global_task_id, global_weights)
            reports.append(rep)
            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, client.task_sequence[global_task_id], loss, train_acc, train_time)

        log_print("  > Step 2: Global Aggregation (Sylva)")
        global_lora, agg_time, mem_start, mem_end = server.aggregate(reports)
        recorder.output_server_agg(agg_time)

        if global_lora:
            comm_cost_per_round = get_model_param_size(global_lora) * 2
            recorder.comm_cost_per_round = comm_cost_per_round

        if args.record_memory:
            recorder.update_memory_stats(np.max(client_mem_peaks), np.mean(client_mem_ends), mem_end, mem_end)
        recorder.update_time_stats(np.sum(client_train_times), agg_time)

        is_last_round_of_task = ((rnd + 1) % args.t_round == 0) or (rnd == args.num_rounds - 1)

        log_print("  > Step 3: Evaluation Across All Learned Tasks")

        all_results = {'clean': [], 'robust': []}
        per_task_results = {'clean': [[] for _ in range(global_task_id + 1)], 'robust': [[] for _ in range(global_task_id + 1)]}

        retrain_times = []
        retrain_mem_peaks = []
        retrain_mem_ends = []

        for client in clients:
            client.update_lora_params(global_lora)
            if is_last_round_of_task:
                new_weights, loss, acc, retrain_time, mem_start, mem_peak, mem_end = client.Sylva_retrain(global_task_id)
                client.set_weights(new_weights)
                retrain_times.append(retrain_time)
                retrain_mem_peaks.append(mem_peak)
                retrain_mem_ends.append(mem_end)
                log_print(f"    Client {client.cid}: Second Stage, Loss={loss:.4f}, Acc={acc:.2f}, Time={retrain_time:.2f}s")
            else:
                retrain_times.append(0)
                retrain_mem_peaks.append(0)
                retrain_mem_ends.append(0)
            
            res = client.extensive_test(rnd)
            for k in all_results:
                all_results[k].append(np.mean(res[k]))
            for tid in range(len(res['clean'])):
                per_task_results['clean'][tid].append(res['clean'][tid])
                per_task_results['robust'][tid].append(res['robust'][tid])

            recorder.output_client_eval(client.cid, res)
            recorder.check_comm_cost_target(global_task_id, res['robust'][global_task_id], rnd)

        recorder.record_task_accuracy(global_task_id, 
            [np.mean(per_task_results['clean'][tid]) for tid in range(global_task_id + 1)],
            [np.mean(per_task_results['robust'][tid]) for tid in range(global_task_id + 1)])

        round_acc_clean.append(np.mean(all_results['clean']))
        round_acc_robust.append(np.mean(all_results['robust']))

        avg_retrain_time = np.sum(retrain_times)
        avg_retrain_mem_peak = np.max(retrain_mem_peaks) if retrain_mem_peaks and any(p > 0 for p in retrain_mem_peaks) else 0
        avg_retrain_mem_end = np.mean([e for e in retrain_mem_ends if e > 0]) if retrain_mem_ends else 0

        if is_last_round_of_task:
            recorder.update_sylva_retrain_stats(avg_retrain_time, avg_retrain_mem_peak, avg_retrain_mem_end)
        
        recorder.output_round_summary(rnd, all_results, avg_retrain_time, avg_retrain_mem_peak, avg_retrain_mem_end)

    total_cost = time.time() - total_start_time
    
    recorder.output_summary(round_acc_clean, round_acc_robust, global_task_id, total_cost, output_dir)


if __name__ == '__main__':
    main()