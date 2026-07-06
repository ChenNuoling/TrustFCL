import torch
import copy
import torch.nn.functional as F
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None
        self.k = min(args.k_sim_clients, args.num_clients - 1)  # 用于聚合的最相似客户端数量

    def _compute_model_similarity(self, weights1, weights2):
        """
        计算两个模型参数的余弦相似度
        参数:
            weights1, weights2: 模型的state_dict
        返回:
            similarity: 余弦相似度值
        """
        vec1 = []
        vec2 = []
        for key in weights1.keys():
            if 'weight' in key or 'bias' in key:
                vec1.append(weights1[key].view(-1))
                vec2.append(weights2[key].view(-1))
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        
        if torch.norm(vec1) == 0 or torch.norm(vec2) == 0:
            return 0.0
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    def _compute_similarity_matrix(self, client_weights):
        """
        预先计算所有客户端之间的相似度矩阵（避免重复计算）
        参数:
            client_weights: dict {client_id: weights}
        返回:
            sim_matrix: dict {cid_i: [(cid_j, sim), ...]} 每个客户端与其他客户端的相似度列表
        """
        client_ids = list(client_weights.keys())
        sim_matrix = {}
        
        for cid_i in client_ids:
            weights_i = client_weights[cid_i]
            similarities = []
            for cid_j in client_ids:
                if cid_i == cid_j:
                    continue
                weights_j = client_weights[cid_j]
                sim = self._compute_model_similarity(weights_i, weights_j)
                similarities.append((cid_j, sim))
            sim_matrix[cid_i] = similarities
        
        return sim_matrix

    def aggregate(self, reports, client_saliency_maps=None):
        """
        SM_v3: 统一聚合方法，同时处理模型参数和SM聚合（避免重复计算相似度）
        参数:
            reports: 包含 state_dict 的列表
            client_saliency_maps: dict {client_id: sm} (可选)
        返回:
            若提供client_saliency_maps: (aggregated_weights_dict, aggregated_sms, agg_time, mem_start, mem_end)
            否则: (aggregated_weights_dict, agg_time, mem_start, mem_end)
            aggregated_weights_dict: dict {client_id: weights} 每个客户端独有的聚合模型
        逻辑:
            1. 模型参数：基于余弦相似度分组聚合，每个客户端i只与最相似的k个客户端聚合
            2. SM聚合：基于余弦相似度分组，每个客户端i只与最相似的k个客户端聚合
            3. 预先计算相似度矩阵，模型参数和SM聚合时复用
        """
        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        if not reports:
            agg_time = time.time() - start_time
            mem_end = get_memory_usage() if self.args.record_memory else 0
            return self.global_weights, agg_time, mem_start, mem_end

        client_ids = list(range(len(reports)))
        client_weights = {i: reports[i] for i in client_ids}
        n = len(client_ids)
        
        if n <= 1:
            aggregated_weights_dict = {cid: copy.deepcopy(reports[cid]) for cid in client_ids}
            self.global_weights = aggregated_weights_dict[0] if n == 1 else None
            
            agg_time = time.time() - start_time
            mem_end = get_memory_usage() if self.args.record_memory else 0
            
            if client_saliency_maps is not None and len(client_saliency_maps) > 0:
                aggregated_sms = {cid: sm for cid, sm in client_saliency_maps.items()}
                return aggregated_weights_dict, aggregated_sms, agg_time, mem_start, mem_end
            
            return aggregated_weights_dict, agg_time, mem_start, mem_end
        
        sim_matrix = self._compute_similarity_matrix(client_weights)
        
        aggregated_weights_dict = {}
        
        for cid_i in client_ids:
            similarities = sim_matrix[cid_i]
            similarities.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [cid for cid, _ in similarities[:self.k]]
            selected_clients.append(cid_i)
            
            agg_w = copy.deepcopy(reports[cid_i])
            for key in agg_w.keys():
                agg_w[key] = torch.zeros_like(agg_w[key]) if isinstance(agg_w[key], torch.Tensor) else 0.0
                
                for cid_j in selected_clients:
                    w_j = reports[cid_j][key]
                    if isinstance(agg_w[key], torch.Tensor):
                        agg_w[key] += w_j / len(selected_clients)
                    else:
                        agg_w[key] += w_j / len(selected_clients)
            
            aggregated_weights_dict[cid_i] = agg_w
        
        self.global_weights = None

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        if client_saliency_maps is not None and len(client_saliency_maps) > 0:
            aggregated_sms = {}
            sm_client_ids = list(client_saliency_maps.keys())
            
            for cid_i in sm_client_ids:
                sm_i = client_saliency_maps[cid_i]
                diff_sum = torch.zeros_like(sm_i)
                count = 0
                
                similarities = sim_matrix[cid_i]
                similarities.sort(key=lambda x: x[1], reverse=True)
                selected_clients = [cid for cid, _ in similarities[:self.k]]
                
                for cid_j in selected_clients:
                    sm_j = client_saliency_maps[cid_j]
                    diff = sm_i - sm_j
                    mask = (diff > 0).float()
                    diff_masked = diff * mask
                    diff_sum += diff_masked
                    count += 1
                
                if count > 0:
                    sm_agg = diff_sum / count
                else:
                    sm_agg = torch.zeros_like(sm_i)
                
                aggregated_sms[cid_i] = sm_agg
            
            return aggregated_weights_dict, aggregated_sms, agg_time, mem_start, mem_end
        
        return aggregated_weights_dict, agg_time, mem_start, mem_end
