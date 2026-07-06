import torch
import copy
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None

    def aggregate(self, reports):
        """
        标准 FedAvg 聚合逻辑
        Args:
            reports: 包含 state_dict 的列表
        Returns:
            聚合后的全局 state_dict, 聚合时间, 聚合前内存, 聚合后内存
        """
        if not reports:
            return self.global_weights, 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        updated_weights = copy.deepcopy(reports[0])

        # 遍历 state_dict 中的所有 key 进行平均
        for key in updated_weights.keys():
            for i in range(1, len(reports)):
                updated_weights[key] += reports[i][key]

            # 处理张量类型或浮点类型
            if isinstance(updated_weights[key], torch.Tensor):
                updated_weights[key] = torch.div(updated_weights[key], len(reports))
            else:
                updated_weights[key] = updated_weights[key] / len(reports)

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        self.global_weights = updated_weights
        return updated_weights, agg_time, mem_start, mem_end
