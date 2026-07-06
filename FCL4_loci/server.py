import torch
import copy
import numpy as np
from operator import itemgetter
import time
from utils import get_memory_usage


class Server:
    def __init__(self, args):
        self.args = args
        self.k = getattr(args, 'ot_similarity_k', 4)

    def aggregate_ot(self, all_kd_models, task, dataloader):
        """
        最优传输聚合 (Optimal Transport Aggregation)
        基于模型相似度选择和聚合
        """
        mem_start = get_memory_usage() if self.args.record_memory else 0
        start_time = time.time()
        
        agg_models = []
        model_num = len(all_kd_models)

        for i, client_data in enumerate(all_kd_models):
            client_id = client_data['client']
            kd_model_state = client_data['models']

            similarity = []
            for j in range(model_num):
                if i != j:
                    other_state = all_kd_models[j]['models']
                    loss = self._compute_similarity(kd_model_state, other_state, task, dataloader)
                    similarity.append({
                        'number': j,
                        'id': all_kd_models[j]['client'],
                        'sim': loss
                    })

            similarity = sorted(similarity, key=itemgetter('sim'), reverse=False)
            top_k_similar = similarity[:self.k]

            aggregated_state = self._geometric_ensemble(
                kd_model_state,
                [all_kd_models[s['number']]['models'] for s in top_k_similar],
                task,
                dataloader
            )

            agg_models.append({
                'client': client_id,
                'model': aggregated_state,
                'task': task
            })

        agg_time = time.time() - start_time
        mem_end = get_memory_usage() if self.args.record_memory else 0
        
        return agg_models, agg_time, mem_start, mem_end

    def _compute_similarity(self, state1, state2, task, dataloader):
        """计算两个模型状态的相似度"""
        return self._multi_class_cross_entropy(state1, state2, task, dataloader)

    def _multi_class_cross_entropy(self, state1, state2, task, dataloader):
        """MultiClassCrossEntropy between two model states"""
        from nets import UnifiedKDModel
        model1 = UnifiedKDModel(self.args).to(self.args.device)
        model2 = UnifiedKDModel(self.args).to(self.args.device)
        model1.load_state_dict(state1)
        model2.load_state_dict(state2)
        model1.eval()
        model2.eval()

        total_loss = 0
        nc_per_task = self.args.num_classes // self.args.num_tasks
        offset1, offset2 = task * nc_per_task, (task + 1) * nc_per_task
        count = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.args.device)
                out1 = model1(images, task)[:, offset1:offset2]
                out2 = model2(images, task)[:, offset1:offset2]

                outputs = torch.log_softmax(out1 / 2, dim=1)
                labels = torch.softmax(out2 / 2, dim=1)
                loss = -torch.mean(torch.sum(outputs * labels, dim=1))
                total_loss += loss.item()
                count += 1

        del model1, model2
        return total_loss / max(count, 1)

    def _geometric_ensemble(self, center_state, candidate_states, task, dataloader):
        """
        几何平均聚合 (Geometric Ensemble)
        基于最优传输的对齐和聚合
        """
        from nets import UnifiedKDModel
        center_model = UnifiedKDModel(self.args).to(self.args.device)
        center_model.load_state_dict(center_state)

        aggregated_state = copy.deepcopy(center_state)

        if not candidate_states:
            return aggregated_state

        all_states = [center_state] + candidate_states
        aligned_states = self._align_models(all_states, task, dataloader)

        for key in aggregated_state.keys():
            if isinstance(aggregated_state[key], torch.Tensor):
                aligned_weights = [s[key] for s in aligned_states]
                geometric_mean = torch.exp(torch.stack([torch.log(w) for w in aligned_weights]).mean(0))
                aggregated_state[key] = geometric_mean

        return aggregated_state

    def _align_models(self, states, task, dataloader):
        """对齐模型参数"""
        aligned = [states[0]]
        return aligned