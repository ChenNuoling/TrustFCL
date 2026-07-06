import torch
import copy
import numpy as np
from scipy.spatial import KDTree
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None

    def _flatten_params(self, param_dict):
        """将参数字典展平为向量"""
        flat = []
        for k in sorted(param_dict.keys()):
            flat.append(param_dict[k].view(-1).cpu().numpy())
        return np.concatenate(flat)

    def _unflatten_params(self, flat_vec, param_template):
        """将向量还原为参数字典"""
        params = copy.deepcopy(param_template)
        idx = 0
        for k in sorted(params.keys()):
            shape = params[k].shape
            numel = params[k].numel()
            params[k] = torch.tensor(flat_vec[idx:idx+numel], dtype=params[k].dtype, device=params[k].device).view(shape)
            idx += numel
        return params

    def aggregate(self, reports):
        """
        Sylva聚合逻辑：
        - 构建球树索引
        - 找相似的LoRA向量
        - 高斯加权聚合LoRA参数
        - 返回全局LoRA参数（不是完整模型权重）
        """
        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        if not reports:
            agg_time = time.time() - start_time
            mem_end = get_memory_usage() if self.args.record_memory else 0
            return self.global_lora if hasattr(self, 'global_lora') else None, agg_time, mem_start, mem_end
        
        # 只对LoRA参数做聚合
        lora_keys = [k for k in reports[0].keys() if 'lora' in k.lower()]
        
        if len(lora_keys) == 0:
            agg_time = time.time() - start_time
            mem_end = get_memory_usage() if self.args.record_memory else 0
            return None, agg_time, mem_start, mem_end
        
        # 构建所有client的LoRA向量
        client_vectors = []
        client_lora_dicts = []
        for rep in reports:
            lora_dict = {k: rep[k] for k in lora_keys}
            client_lora_dicts.append(lora_dict)
            client_vectors.append(self._flatten_params(lora_dict))
        client_vectors = np.array(client_vectors)
        
        # 使用KDTree替代BallTree
        k = min(self.args.sylva_k_neighbors, len(reports))  # 找最近的k个邻居
        tree = KDTree(client_vectors)
        
        # 聚合后的向量
        aggregated_vec = np.zeros_like(client_vectors[0])
        
        for i in range(len(reports)):
            # 找最近k个邻居（包括自己）
            distances, indices = tree.query([client_vectors[i]], k=k)
            
            # 高斯权重
            sigma = np.max(distances) if np.max(distances) > 0 else 1.0
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            weights = weights / np.sum(weights)  # 归一化
            
            # 加权平均
            weighted_sum = np.zeros_like(client_vectors[0])
            for j in range(len(indices[0])):
                weighted_sum += weights[0][j] * client_vectors[indices[0][j]]
            aggregated_vec += weighted_sum
        
        aggregated_vec = aggregated_vec / len(reports)
        
        # 还原为参数字典
        aggregated_lora = self._unflatten_params(aggregated_vec, client_lora_dicts[0])
        
        self.global_lora = aggregated_lora

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        return aggregated_lora, agg_time, mem_start, mem_end
