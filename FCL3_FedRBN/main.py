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
FedRBN 方法实现：
1、每一轮都固定比例（如20%）将用户（客户端）随机分为AT用户和ST用户；
2、每个用户维护双分支批归一化（Dual BN），BN_c（用于干净样本）和 BN_a（用于对抗样本），
AT用户在训练时同时使用两者，ST用户只训练 BN_c，而 BN_a 冻结；
3、服务器聚合所有非BN参数，BN_c 或 BN_a保持本地化（除了ST的BN_a），
使用余弦相似度基于 BN_c 的均值和方差，计算不同用户之间的分布相似性，
根据相似性对AT用户的 BN_a 进行加权平均，生成ST用户的 BN_a；
4、ST用户在本地训练时，除了标准的干净样本损失外，还会增加一个额外的PNC损失，
将干净样本也通过 BN_a 分支，计算交叉熵损失（无需生成对抗样本），
通过加权系数 λ（默认0.5）平衡标准精度与鲁棒性，L_pnc =(1−λ)⋅ℓ_c +λ⋅ℓ_a
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
    log_print("Initializing FedRBN with Dual BN components...")
    task_gen = TaskGenerator(args)

    dummy_model = Init_model(args)
    server = Server(args)

    all_task_ids = list(range(args.num_tasks))
    client_task_sequences = [np.random.permutation(all_task_ids).tolist() for _ in range(args.num_clients)]
    log_print(f"Client task sequences: {client_task_sequences}")
    clients = [
        Client(i, args, copy.deepcopy(dummy_model), task_gen, task_sequence=client_task_sequences[i])
        for i in range(args.num_clients)
    ]

    log_print("=== FedRBN (Randomized Sequences) ===")

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

        num_AT = max(1, int(args.num_clients * args.AT_ratio))
        at_indices = np.random.choice(args.num_clients, num_AT, replace=False)
        st_indices = np.array([i for i in range(args.num_clients) if i not in at_indices])

        log_print(f"  > AT Clients: {at_indices.tolist()}, ST Clients: {st_indices.tolist()}")

        reports = []
        client_train_times = []
        client_mem_peaks = []
        client_mem_ends = []

        for idx in at_indices:
            client = clients[idx]
            client.is_AT = True
            non_bn, bn, bn_c_mean_var, loss, train_acc, train_time, mem_start, mem_peak, mem_end = client.AT_train_main_task(global_task_id)
            reports.append((non_bn, bn, bn_c_mean_var, idx, True))
            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, client.task_sequence[global_task_id], loss, train_acc, train_time, client_type="AT")

        for idx in st_indices:
            client = clients[idx]
            client.is_AT = False
            non_bn, bn, bn_c_mean_var, loss, train_acc, train_time, mem_start, mem_peak, mem_end = client.ST_train_main_task(global_task_id)
            reports.append((non_bn, bn, bn_c_mean_var, idx, False))
            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, client.task_sequence[global_task_id], loss, train_acc, train_time, client_type="ST")

        log_print("  > Step 2: Global Aggregation")
        global_non_bn, st_bn_a_updates, agg_time, mem_start, mem_end = server.aggregate(reports)
        recorder.output_server_agg(agg_time)

        if comm_cost_per_round == 0 and global_non_bn:
            comm_cost_per_round = get_model_param_size(global_non_bn) * 2
            recorder.comm_cost_per_round = comm_cost_per_round

        if args.record_memory:
            recorder.update_memory_stats(np.max(client_mem_peaks), np.mean(client_mem_ends), mem_end, mem_end)
        recorder.update_time_stats(np.sum(client_train_times), agg_time)

        log_print("  > Step 3: Update Client Models")
        for idx in at_indices:
            clients[idx].set_AT_weights(global_non_bn)

        for idx in st_indices:
            bn_a_update = st_bn_a_updates.get(idx, {})
            clients[idx].set_ST_weights(global_non_bn, bn_a_update)

        log_print("  > Step 4: Evaluation Across All Learned Tasks")

        all_results = {'clean': [], 'robust': []}
        per_task_results = {'clean': [[] for _ in range(global_task_id + 1)], 'robust': [[] for _ in range(global_task_id + 1)]}

        for client in clients:
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
        recorder.output_round_summary(rnd, all_results)

    total_cost = time.time() - total_start_time
    
    recorder.output_summary(round_acc_clean, round_acc_robust, global_task_id, total_cost, output_dir)


if __name__ == '__main__':
    main()