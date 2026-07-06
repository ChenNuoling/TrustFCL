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
Naive Federated Continual Learning (FCL) 流程
特征：标准 FedAvg + 顺序任务学习，没有任何遗忘缓解机制 (No FedWeIT, No EWC, No Replay)
增强：每个客户端拥有独立的随机任务序列，模拟更复杂的异构持续学习场景。


示例：
- 全局任务ID=2，客户端1训练任务4（标签范围[30,40)），映射后标签范围[0,10)，模型选择[20,30)分支
- 全局任务ID=2，客户端2训练任务9（标签范围[80,90)），映射后标签范围[0,10)，模型选择[20,30)分支


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
    log_print("Initializing Plain FCL components with Randomized Task Sequences...")
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

    log_print("=== Federated Continual Learning (Randomized Sequences) ===")

    total_start_time = time.time()
    round_acc_clean = []
    round_acc_robust = []

    prev_global_task_id = -1

    comm_cost_per_round = get_model_param_size(global_weights) * 2
    recorder = Recorder(log_print, args, comm_cost_per_round)

    for rnd in range(args.num_rounds):
        recorder.output_round_start(rnd)

        global_task_id = min(rnd // args.t_round, args.num_tasks - 1)

        if global_task_id != prev_global_task_id and prev_global_task_id >= 0:
            recorder.output_task_switch(prev_global_task_id, global_task_id)
        prev_global_task_id = global_task_id

        log_print("  > Step 1: Standard Local Training")
        reports = []
        client_train_times = []
        client_mem_peaks = []
        client_mem_ends = []

        for client in clients:
            rep, loss, train_acc, train_time, mem_start, mem_peak, mem_end = client.pgd_train_main_task(global_task_id, global_weights)
            reports.append(rep)
            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, client.task_sequence[global_task_id], loss, train_acc, train_time)

        global_weights, agg_time, mem_start, mem_end = server.aggregate(reports)
        recorder.output_server_agg(agg_time)

        if args.record_memory:
            recorder.update_memory_stats(np.max(client_mem_peaks), np.mean(client_mem_ends), mem_end, mem_end)
        recorder.update_time_stats(np.sum(client_train_times), agg_time)

        log_print("  > Step 3: Evaluation Across All Learned Tasks")

        all_results = {'clean': [], 'robust': []}
        per_task_results = {'clean': [[] for _ in range(global_task_id + 1)], 'robust': [[] for _ in range(global_task_id + 1)]}

        for client in clients:
            client.set_weights(global_weights)
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
