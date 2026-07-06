import torch
import torch.nn.functional as F
import copy
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None

    def _compute_bn_c_similarity(self, mean_var1, mean_var2):
        total_sim = 0.0
        count = 0
        for key in mean_var1:
            if key in mean_var2:
                m1, v1 = mean_var1[key]
                m2, v2 = mean_var2[key]
                if isinstance(m1, torch.Tensor) and isinstance(m2, torch.Tensor):
                    m1 = m1.flatten()
                    v1 = v1.flatten()
                    m2 = m2.flatten()
                    v2 = v2.flatten()
                    vec1 = torch.cat([m1, v1])
                    vec2 = torch.cat([m2, v2])
                    sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                else:
                    sim = 1.0
                total_sim += sim
                count += 1
        return total_sim / count if count > 0 else 0.0

    def aggregate(self, reports):
        """
        FedRBN聚合逻辑
        Args:
            reports: 包含 (non_bn_params, bn_params, bn_c_mean_var, client_idx, is_AT, training_time, mem_start, mem_peak, mem_end) 的列表
        Returns:
            (global_non_bn, st_bn_a_updates, agg_time, mem_start, mem_end)
        """
        if not reports:
            return self.global_weights, {}, 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        non_bn_reports = [r[0] for r in reports]
        bn_reports = [r[1] for r in reports]
        bn_c_mean_vars = [r[2] for r in reports]
        client_indices = [r[3] for r in reports]
        is_AT_list = [r[4] for r in reports]

        global_non_bn = copy.deepcopy(non_bn_reports[0])
        for key in global_non_bn.keys():
            for i in range(1, len(non_bn_reports)):
                global_non_bn[key] += non_bn_reports[i][key]
            if isinstance(global_non_bn[key], torch.Tensor):
                global_non_bn[key] = torch.div(global_non_bn[key], len(non_bn_reports))
            else:
                global_non_bn[key] /= len(non_bn_reports)

        at_indices = [i for i, is_AT in enumerate(is_AT_list) if is_AT]
        st_indices = [i for i, is_AT in enumerate(is_AT_list) if not is_AT]

        st_bn_a_updates = {}
        if at_indices and st_indices:
            bn_a_keys = [k for k in bn_reports[0].keys() if 'bn_a' in k]
            
            for st_idx in st_indices:
                st_client_idx = client_indices[st_idx]
                st_bn_c = bn_c_mean_vars[st_idx]
                
                similarities = []
                for at_idx in at_indices:
                    at_bn_c = bn_c_mean_vars[at_idx]
                    sim = self._compute_bn_c_similarity(st_bn_c, at_bn_c)
                    similarities.append((at_idx, max(sim, 0.0)))
                
                total_sim = sum(s[1] for s in similarities)
                if total_sim > 0:
                    st_bn_a_updates[st_client_idx] = {}
                    for key in bn_a_keys:
                        weighted_sum = None
                        for at_idx, sim in similarities:
                            weight = sim / total_sim
                            at_bn_a = bn_reports[at_idx].get(key)
                            if at_bn_a is not None:
                                if weighted_sum is None:
                                    weighted_sum = at_bn_a.detach().clone() * weight
                                else:
                                    weighted_sum += at_bn_a.detach().clone() * weight
                        if weighted_sum is not None:
                            st_bn_a_updates[st_client_idx][key] = weighted_sum

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        self.global_weights = global_non_bn
        return global_non_bn, st_bn_a_updates, agg_time, mem_start, mem_end
