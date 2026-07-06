import torch
import copy
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.global_weights = None
        self.global_bn = None

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
    
    def aggregate_bn(self, bn_reports, adv_losses):
        """
        Fcat BN层参数聚合：根据对抗损失大小加权平均
        对抗损失越大，权重越大（表示该客户端需要更多关注）
        
        Args:
            bn_reports: 包含BN层参数的字典列表
            adv_losses: 各客户端的对抗损失列表
            
        Returns:
            聚合后的全局BN参数, 聚合时间, 聚合前内存, 聚合后内存
        """
        if not bn_reports:
            return self.global_bn, 0, 0, 0

        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()

        # 根据对抗损失计算权重（损失越大权重越大）
        # 使用softmax归一化权重
        losses_tensor = torch.tensor(adv_losses, dtype=torch.float32)
        weights = torch.softmax(losses_tensor, dim=0).tolist()

        # 初始化聚合结果
        aggregated_bn = copy.deepcopy(bn_reports[0])
        
        # 遍历所有BN参数key
        for key in aggregated_bn.keys():
            # 使用加权平均
            aggregated_bn[key] = weights[0] * aggregated_bn[key]
            for i in range(1, len(bn_reports)):
                if key in bn_reports[i]:
                    aggregated_bn[key] += weights[i] * bn_reports[i][key]

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0

        self.global_bn = aggregated_bn
        return aggregated_bn, agg_time, mem_start, mem_end
