from utils import Args, set_seed, get_model_param_size, LogPrinter, Recorder
from nets import Init_model
from data_loader import TaskGenerator
from client import Client
from server import Server
import numpy as np
import time
import copy
import os
from datetime import datetime

"""
Federated Continual Learning (FCL) with EWC + KD + OT
特征：EWC (Elastic Weight Consolidation) + Knowledge Distillation + Optimal Transport Aggregation
增强：每个客户端拥有独立的随机任务序列，支持KD模型剪枝和OT相似度聚合。

5/18 15:00更新FL_Avg_ex\FCL\FCL_loci
核心逻辑：
- 全局任务ID (task_idx_in_seq): 当前训练的第几个任务，所有客户端相同，用于选择分类器分支
- 本地真实任务ID (actual_task_id): 数据集中对应的真实任务ID，各客户端不同，用于标签映射

client.py: 将全局标签映射为本地标签，映射公式：local_label = global_label - actual_task_id * per_task_class_num
net.py: 根据全局任务ID选择分类器分支范围，例如全局任务ID为2，选择[per_task_class_num*2:per_task_class_num*3)范围

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
    log_print("Initializing FCL components with Randomized Task Sequences...")
    log_print(">>> Using EWC + KD (Knowledge Distillation) + OT (Optimal Transport) for continual learning <<<")
    log_print(">>> No global weights, only OT aggregation <<<")

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

    log_print("=== Federated Continual Learning (EWC + KD + OT) ===")

    total_start_time = time.time()
    round_acc_clean = []
    round_acc_robust = []
    
    # 存储聚合后的 KD 模型（参考 main_EWC.py）
    w_globals = []

    comm_cost_per_round = 0
    recorder = Recorder(log_print, args, comm_cost_per_round)

    for rnd in range(args.num_rounds):
        recorder.output_round_start(rnd)

        task_idx_in_seq = min(rnd // args.t_round, args.num_tasks - 1)

        log_print(f"  > Step 1: Local Training (Global Task Index {task_idx_in_seq})")
        all_kd_models = []
        client_train_times = []
        client_mem_peaks = []
        client_mem_ends = []

        for client in clients:
            actual_task_id = client.task_sequence[task_idx_in_seq]

            if len(w_globals) != 0:
                agg_client_ids = [w['client'] for w in w_globals]
                if client.cid in agg_client_ids:
                    client.cur_kd.load_state_dict(w_globals[agg_client_ids.index(client.cid)]['model'])

            rep, loss, train_acc, train_time, mem_start, mem_peak, mem_end = client.pgd_train_ewc_kd_task(task_idx_in_seq, None)

            loader, _ = client.data_generator.get_loader(client.cid, actual_task_id, 'train')
            kd_state = client.prune_kd_model(loader, actual_task_id)
            all_kd_models.append({'client': client.cid, 'models': kd_state['model'], 'task': actual_task_id})

            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, actual_task_id, loss, train_acc, train_time)

        log_print("  > Step 2: OT Aggregation")
        agg_time = 0
        server_mem_start, server_mem_end = 0, 0
        if rnd % args.t_round == args.t_round - 1:
            w_globals = []
            log_print(f"    Global Task {task_idx_in_seq} completed, resetting aggregated KD models")
        else:
            first_client_task_id = clients[0].task_sequence[task_idx_in_seq]
            loader, _ = clients[0].data_generator.get_loader(0, first_client_task_id, 'test')
            w_globals, agg_time, server_mem_start, server_mem_end = server.aggregate_ot(all_kd_models, first_client_task_id, loader)
            log_print(f"    Using OT Aggregation, {len(w_globals)} clients updated")

        recorder.output_server_agg(agg_time)

        if comm_cost_per_round == 0 and len(all_kd_models) > 0:
            comm_cost_per_round = len(all_kd_models) * 1024 * 1024  # 估算值
            recorder.comm_cost_per_round = comm_cost_per_round

        if args.record_memory:
            recorder.update_memory_stats(np.max(client_mem_peaks), np.mean(client_mem_ends), server_mem_start, server_mem_end)
        recorder.update_time_stats(np.sum(client_train_times), agg_time)

        log_print("  > Step 3: Evaluation Across All Learned Tasks")
        all_results = {'clean': [], 'robust': []}
        per_task_results = {'clean': [[] for _ in range(task_idx_in_seq + 1)], 'robust': [[] for _ in range(task_idx_in_seq + 1)]}

        for client in clients:
            res = client.extensive_test(rnd)
            for k in all_results:
                all_results[k].append(np.mean(res[k]))
            for tid in range(len(res['clean'])):
                per_task_results['clean'][tid].append(res['clean'][tid])
                per_task_results['robust'][tid].append(res['robust'][tid])

            recorder.output_client_eval(client.cid, res)
            if len(res['robust']) > task_idx_in_seq:
                recorder.check_comm_cost_target(task_idx_in_seq, res['robust'][task_idx_in_seq], rnd)

        if len(res['robust']) > 0:
            recorder.record_task_accuracy(task_idx_in_seq, 
                [np.mean(per_task_results['clean'][tid]) for tid in range(task_idx_in_seq + 1)],
                [np.mean(per_task_results['robust'][tid]) for tid in range(task_idx_in_seq + 1)])

        round_acc_clean.append(np.mean(all_results['clean']))
        round_acc_robust.append(np.mean(all_results['robust']))
        recorder.output_round_summary(rnd, all_results)

    total_cost = time.time() - total_start_time
    
    recorder.output_summary(round_acc_clean, round_acc_robust, task_idx_in_seq, total_cost, output_dir)


if __name__ == '__main__':
    main()
