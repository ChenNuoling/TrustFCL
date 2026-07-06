import torch
import copy
import numpy as np
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None
        self.alpha = getattr(args, 'slack_alpha', 0.1)  
        self.k = getattr(args, 'slack_k', None)  

    def aggregate(self, reports, losses=None):
        """
        Slack聚合逻辑：基于损失的客户端权重调整
        Args:
            reports: 包含 state_dict 的列表
            losses: 每个客户端的损失值列表（可选）
        Returns:
            聚合后的全局 state_dict, 聚合时间, 聚合前内存, 聚合后内存
        """
        if not reports:
            return self.global_weights, 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        num_clients = len(reports)
        
        if losses is not None and len(losses) == num_clients:
            k = self.k if self.k is not None else num_clients // 2
            k = max(1, min(k, num_clients))
            
            loss_indices = np.argsort(losses)
            top_k_indices = set(loss_indices[:k])
            
            weights = []
            for i in range(num_clients):
                if i in top_k_indices:
                    weights.append(1 + self.alpha)
                else:
                    weights.append(1 - self.alpha)
            
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / num_clients] * num_clients

        updated_weights = copy.deepcopy(reports[0])
        for key in updated_weights.keys():
            updated_weights[key] = updated_weights[key] * weights[0]
            for i in range(1, num_clients):
                updated_weights[key] += reports[i][key] * weights[i]

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        self.global_weights = updated_weights
        return updated_weights, agg_time, mem_start, mem_end
