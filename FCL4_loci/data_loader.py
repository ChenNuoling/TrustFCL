import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict


class TaskGenerator:
    def __init__(self, args):
        self.args = args
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # 删除归一化，因为在进行 PGD 时，通常要求输入图像处于原始像素空间
            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # 加载数据
        print("Loading CIFAR-100...")
        self.train_set = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_set = datasets.CIFAR100(
            root=args.data_root,
            train=False,
            download=True,
            transform=self.transform
        )

        # 按类分组索引
        self.train_indices_by_class = self._group_by_class(self.train_set)
        self.test_indices_by_class = self._group_by_class(self.test_set)

        # 定义每个任务的类别范围
        self.classes_per_task = 100 // args.num_tasks
        self.tasks_config = []
        all_classes = np.arange(100)
        for i in range(args.num_tasks):
            start = i * self.classes_per_task
            end = (i + 1) * self.classes_per_task
            self.tasks_config.append(all_classes[start:end].tolist())

        # 为每个任务内的每个 client 生成 Dirichlet 分布的数据分配
        print("\nGenerating Dirichlet distribution for each task...")
        self.client_data_map = self._generate_dirichlet_split_per_task()

        # 验证生成成功
        if not self.client_data_map:
            raise RuntimeError("Failed to generate client_data_map")
        print("✓ TaskGenerator initialized successfully\n")

    def _group_by_class(self, dataset):
        """按类别分组样本索引"""
        indices = defaultdict(list)
        for idx, label in enumerate(dataset.targets):
            indices[label].append(idx)
        # 打乱每个类别的索引
        for cls in indices:
            np.random.shuffle(indices[cls])
        return indices

    def _generate_dirichlet_split_per_task(self):
        """
        为每个任务内的每个 client 生成 Dirichlet 分布的数据分配（限制每个client的数据量）
        返回: {
            task_id: {
                client_id: {
                    'train_indices': [...],
                    'test_indices': [...]
                }
            }
        }
        """
        num_clients = self.args.num_clients
        num_tasks = self.args.num_tasks
        alpha = self.args.dirichlet_alpha

        # 设置每个 client 的最大样本数限制（默认 2000）
        max_train_samples = getattr(self.args, 'max_train_samples_per_client', 2000)
        max_test_samples = getattr(self.args, 'max_test_samples_per_client', 500)

        client_data_map = {}

        for task_id, task_classes in enumerate(self.tasks_config):
            print(f"\nProcessing Task {task_id}: Classes {task_classes[0]}-{task_classes[-1]}")

            num_classes_in_task = len(task_classes)

            # 为该任务的每个类别生成 Dirichlet 分布
            # shape: (num_classes_in_task, num_clients)
            class_proportions = np.random.dirichlet(
                [alpha] * num_clients,
                size=num_classes_in_task
            )

            task_data = {}

            # 初始化每个 client 的数据列表
            for client_id in range(num_clients):
                task_data[client_id] = {
                    'train_indices': [],
                    'test_indices': []
                }

            # 为每个类别分配样本
            for class_idx, class_id in enumerate(task_classes):
                # 获取该类别的所有训练/测试样本索引
                class_train_idx = self.train_indices_by_class[class_id].copy()
                class_test_idx = self.test_indices_by_class[class_id].copy()

                # 打乱顺序，确保随机分配
                np.random.shuffle(class_train_idx)
                np.random.shuffle(class_test_idx)

                # 记录当前分配到每个 client 的索引位置
                train_start_idx = 0
                test_start_idx = 0

                # 为每个 client 分配该类别的样本
                for client_id in range(num_clients):
                    proportion = class_proportions[class_idx][client_id]

                    # 计算理论分配数量
                    theoretical_train = int(len(class_train_idx) * proportion)
                    theoretical_test = int(len(class_test_idx) * proportion)

                    # 检查该 client 当前已有的样本数
                    current_train_count = len(task_data[client_id]['train_indices'])
                    current_test_count = len(task_data[client_id]['test_indices'])

                    # 限制训练样本数量
                    remaining_train_capacity = max_train_samples - current_train_count
                    if remaining_train_capacity <= 0:
                        num_train_samples = 0
                    else:
                        num_train_samples = min(theoretical_train, remaining_train_capacity)

                    # 限制测试样本数量
                    remaining_test_capacity = max_test_samples - current_test_count
                    if remaining_test_capacity <= 0:
                        num_test_samples = 0
                    else:
                        num_test_samples = min(theoretical_test, remaining_test_capacity)

                    # 分配训练样本（按顺序分配，避免重复）
                    if num_train_samples > 0 and train_start_idx < len(class_train_idx):
                        end_idx = min(train_start_idx + num_train_samples, len(class_train_idx))
                        assigned_train = class_train_idx[train_start_idx:end_idx]
                        task_data[client_id]['train_indices'].extend(assigned_train)
                        train_start_idx = end_idx

                    # 分配测试样本
                    if num_test_samples > 0 and test_start_idx < len(class_test_idx):
                        end_idx = min(test_start_idx + num_test_samples, len(class_test_idx))
                        assigned_test = class_test_idx[test_start_idx:end_idx]
                        task_data[client_id]['test_indices'].extend(assigned_test)
                        test_start_idx = end_idx

                    # 如果已经达到容量上限，跳过后续分配
                    if len(task_data[client_id]['train_indices']) >= max_train_samples:
                        # 这个 client 已经满了，不再分配更多样本
                        pass

            # 打乱每个 client 的数据顺序
            for client_id in range(num_clients):
                np.random.shuffle(task_data[client_id]['train_indices'])
                np.random.shuffle(task_data[client_id]['test_indices'])

            client_data_map[task_id] = task_data

            # 打印该任务的分布统计
            self._print_task_distribution_stats(task_id, task_classes, class_proportions, task_data)

        return client_data_map

    def _print_task_distribution_stats(self, task_id, task_classes, class_proportions, task_data):
        """打印某个任务内每个 client 的类别分布统计"""
        num_clients = self.args.num_clients
        num_classes_in_task = len(task_classes)

        print(f"\n  Task {task_id} Statistics (alpha={self.args.dirichlet_alpha}):")
        print(f"  Classes: {task_classes[0]}-{task_classes[-1]}")

        # 每个 client 在该任务中拥有的类别数（非零样本的类别）
        classes_per_client = []
        for client_id in range(num_clients):
            nonzero_classes = 0
            for class_idx in range(num_classes_in_task):
                if class_proportions[class_idx][client_id] > 0.01:  # 比例 > 1%
                    nonzero_classes += 1
            classes_per_client.append(nonzero_classes)

        print(f"    Avg classes per client: {np.mean(classes_per_client):.2f}")
        print(f"    Std classes per client: {np.std(classes_per_client):.2f}")
        print(f"    Min/Max classes per client: {np.min(classes_per_client)}/{np.max(classes_per_client)}")

        # 每个 client 在该任务中的总样本数
        train_samples_per_client = [
            len(task_data[cid]['train_indices'])
            for cid in range(num_clients)
        ]
        test_samples_per_client = [
            len(task_data[cid]['test_indices'])
            for cid in range(num_clients)
        ]

        print(f"    Avg train samples per client: {np.mean(train_samples_per_client):.2f}")
        print(
            f"    Max/Min train samples per client: {np.max(train_samples_per_client)}/{np.min(train_samples_per_client)}")
        print(f"    Avg test samples per client: {np.mean(test_samples_per_client):.2f}")
        print(f"    Total train samples in this task: {sum(train_samples_per_client)}")
        print(f"    Total test samples in this task: {sum(test_samples_per_client)}")

    def _print_task_distribution_stats(self, task_id, task_classes, class_proportions, task_data):
        """打印某个任务内每个 client 的类别分布统计"""
        num_clients = self.args.num_clients
        num_classes_in_task = len(task_classes)

        print(f"\n  Task {task_id} Statistics (alpha={self.args.dirichlet_alpha}):")
        print(f"  Classes: {task_classes[0]}-{task_classes[-1]}")

        # 每个 client 在该任务中拥有的类别数（非零样本的类别）
        classes_per_client = []
        for client_id in range(num_clients):
            nonzero_classes = 0
            for class_idx in range(num_classes_in_task):
                if class_proportions[class_idx][client_id] > 0.01:  # 比例 > 1%
                    nonzero_classes += 1
            classes_per_client.append(nonzero_classes)

        print(f"    Avg classes per client: {np.mean(classes_per_client):.2f}")
        print(f"    Std classes per client: {np.std(classes_per_client):.2f}")
        print(f"    Min/Max classes per client: {np.min(classes_per_client)}/{np.max(classes_per_client)}")

        # 每个 client 在该任务中的总样本数（使用传入的 task_data）
        samples_per_client = [
            len(task_data[cid]['train_indices'])
            for cid in range(num_clients)
        ]
        print(f"    Avg samples per client: {np.mean(samples_per_client):.2f}")
        print(f"    Total train samples in this task: {sum(samples_per_client)}")

    def get_loader(self, client_id, task_id, mode='train'):
        """
        获取指定任务中指定 client 的 DataLoader

        Args:
            task_id: 任务 ID (0 到 num_tasks-1)
            client_id: 客户端 ID (0 到 num_clients-1)
            mode: 'train' 或 'test'

        Returns:
            DataLoader, task_classes (当前任务的类别列表)
        """
        # 添加错误检查
        if not hasattr(self, 'client_data_map'):
            raise AttributeError("client_data_map not initialized. Did __init__ complete successfully?")

        if task_id not in self.client_data_map:
            raise ValueError(f"Task {task_id} not found. Available tasks: {list(self.client_data_map.keys())}")
        if client_id not in self.client_data_map[task_id]:
            raise ValueError(
                f"Client {client_id} not found in Task {task_id}. Available clients: {list(self.client_data_map[task_id].keys())}")

        indices = self.client_data_map[task_id][client_id][f'{mode}_indices']

        # 检查是否有数据
        if len(indices) == 0:
            print(f"Warning: Client {client_id} in Task {task_id} has no {mode} data")
            # 返回空的 DataLoader
            dataset = self.train_set if mode == 'train' else self.test_set
            empty_subset = Subset(dataset, [])
            return DataLoader(empty_subset, batch_size=self.args.batch_size), self.tasks_config[task_id]

        dataset = self.train_set if mode == 'train' else self.test_set
        subset = Subset(dataset, indices)

        shuffle = (mode == 'train')
        loader = DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 2
        )

        # 返回当前任务的类别列表
        task_classes = self.tasks_config[task_id]

        return loader, task_classes
