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
SM方法总结：
1、本地增强显著性图S=(s'_r,s'_g,s'_b)由三个通道的判别性结果堆叠而成,
对于 RGB 三个通道s_r、s_g、s_b分别能量归一化,除以该通道所有像素绝对值之和，得到s'_r,s'_g,s'_b
2、客户端本地当前任务所有图像对应一个平均显著性图，作为任务的SM图，反映模型在该任务上的整体注意力分布
3、服务器端，对于客户端i，遍历其他客户端，S_i-S_j,若S_i>S_j保留差值,否则置为 0,对于每个客户端返回
平局聚合后的模型参数和一个独有的SM_agg
4、在新任务 T+1 中生成对抗样本时，扰动区域被服务器下发的显著图弱限制，或者类似乘以显著图，
做一个增强，若没有则不限制 (SM22)
5、旧任务的SM图S_t−1是训练前由的模型根据当前任务数据生成的，S_t是训练结束后模型生成的SM图，
为了保留旧任务的注意力，取新旧SM的软并集即，逐像素取最大值，作为最终上传客户端的图
6、对抗训练时，新增显著图蒸馏损失（SD-Loss）让SM_t包含SM_agg,即当 SM_t 在某个像素上低于 
SM_agg 在该像素上的值时，才惩罚。

SM_v3：
1、服务器端，对于客户端i，遍历其他客户端,选出模型最相似（余弦相似度)的k个客户端集合C_i_sim,其余不相似的
记为C_i_no_sim。对于C_i_sim中的客户端j的SM图，S_i-S_j,若S_i>S_j保留差值,否则置为 0,模型平均聚合返回给客户端i。对于
对于每个客户端i，返回C_i_sim中的客户端平局聚合后的模型参数和一个独有的SM_agg
2、对抗训练时，新增显著图蒸馏损失（SD-Loss）让SM_t包含SM_agg和S_t−1,即当 SM_t 在某个像素上低于 
SM_agg 或S_t−1 在该像素上的值时，才惩罚。

CalFAT训练：
对抗训练损失函数使用CalFAT的CE损失和CKL损失，其中，CCE损失时让模型关注少数类，
使用logits+log π 类先验概率和对数校准，减少多数类的自信，
CKL损失则是衡量原始样本和对抗样本之间概率分布的差异，最大化 ℓ_ckl，
让对抗样本的预测分布尽可能偏离原始样本的预测分布。
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
    log_print("Initializing SM FCL components with Randomized Task Sequences...")
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

    log_print("=== Federated Continual Learning with SM (Randomized Sequences) ===")

    total_start_time = time.time()
    round_acc_clean = []
    round_acc_robust = []

    prev_global_task_id = -1

    client_agg_weights = None
    prev_sm_aggs = {}

    comm_cost_per_round = 0
    recorder = Recorder(log_print, args, comm_cost_per_round)

    for rnd in range(args.num_rounds):
        recorder.output_round_start(rnd)

        global_task_id = min(rnd // args.t_round, args.num_tasks - 1)

        if global_task_id != prev_global_task_id and prev_global_task_id >= 0:
            recorder.output_task_switch(prev_global_task_id, global_task_id)
        prev_global_task_id = global_task_id

        log_print("  > Step 1: Local Training")
        reports = []
        client_saliency_maps = {}
        client_train_times = []
        client_mem_peaks = []
        client_mem_ends = []

        for client in clients:
            if args.use_calfat_training:
                rep, loss, train_acc, task_sm, train_time, mem_start, mem_peak, mem_end = client.train_main_task_with_calfat(global_task_id)
                client_saliency_maps[client.cid] = task_sm
            else:
                client_weights = client_agg_weights[client.cid] if client_agg_weights else None
                rep, loss, train_acc = client.train_main_task(global_task_id, client_weights)
                train_time, mem_start, mem_peak, mem_end = 0, 0, 0, 0
            
            reports.append(rep)
            client_train_times.append(train_time)
            client_mem_peaks.append(mem_peak)
            client_mem_ends.append(mem_end)
            recorder.output_client_train(client.cid, client.task_sequence[global_task_id], loss, train_acc, train_time)

        log_print("  > Step 2: Global Aggregation")
        sm_aggs = None
        if args.use_calfat_training and len(client_saliency_maps) > 0:
            log_print("  > Step 2: Global Aggregation (with SM_v3)")
            result = server.aggregate(reports, client_saliency_maps)
            client_agg_weights, sm_aggs, agg_time, mem_start, mem_end = result[0], result[1], result[2], result[3], result[4]
            
            for client in clients:
                if client.cid in sm_aggs:
                    client.set_sm_agg(sm_aggs[client.cid])
                if client.cid in client_agg_weights:
                    client.set_weights(client_agg_weights[client.cid])
            prev_sm_aggs = sm_aggs
        else:
            result = server.aggregate(reports)
            client_agg_weights, agg_time, mem_start, mem_end = result[0], result[1], result[2], result[3]
            
            for client in clients:
                if client.cid in client_agg_weights:
                    client.set_weights(client_agg_weights[client.cid])

        recorder.output_server_agg(agg_time)

        if comm_cost_per_round == 0 and client_agg_weights:
            first_cid = list(client_agg_weights.keys())[0]
            comm_cost_per_round = get_model_param_size(client_agg_weights[first_cid]) * 2
            recorder.comm_cost_per_round = comm_cost_per_round

        if args.record_memory:
            recorder.update_memory_stats(np.max(client_mem_peaks), np.mean(client_mem_ends), mem_end, mem_end)
        recorder.update_time_stats(np.sum(client_train_times), agg_time)

        log_print("  > Step 3: Evaluation Across All Learned Tasks")

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