import random
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

class TwoBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last, labeled_bs, labeled_num ):
        # 假设sampler是SequentialSampler，即按顺序采样
        # 你可以根据需要替换为其他类型的sampler
        self.sampler = SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.labeled_bs = labeled_bs
        self.labeled_num = labeled_num

    def __iter__(self):
        # 从sampler获取所有索引
        full_indices = list(self.sampler)
        num_samples = len(full_indices)

        # 计算剩余样本数量，用于确定是否丢弃最后一个不完整的批次
        remainder = num_samples % self.batch_size

        # 如果需要丢弃最后一个不完整的批次
        if self.drop_last and remainder > 0:
            num_samples -= remainder

        # 将索引分为两部分
        first_part, second_part = full_indices[:self.labeled_num], full_indices[self.labeled_num:]
        # 对标注和未标注的索引进行打乱
        random.shuffle(first_part)
        random.shuffle(second_part)
        
        # 创建批次
        batch_indices = []
        # 计算每个epoch中的批次总数
        num_labeled_batch = (len(first_part) // (self.labeled_bs))
        num_unlabeled_batch = (len(second_part) // (self.batch_size - self.labeled_bs))
        # print("!!!!!!")
        # print(num_labeled_batch)
        # print(num_unlabeled_batch)
        num_batches = max(num_labeled_batch,num_unlabeled_batch)
        if not self.drop_last and len(self.sampler) % self.batch_size != 0:
            num_batches -= 1
        for i in range(num_batches):
            # 从第一部分中取样本
            start_idx_labeled = i % num_labeled_batch * self.labeled_bs
            end_idx_labeled = start_idx_labeled + self.labeled_bs
            batch_from_first_part = first_part[start_idx_labeled:end_idx_labeled]

            # 从第二部分中取样本，补足一个批次
            start_idx_unlabeled = i % num_unlabeled_batch * (self.batch_size - self.labeled_bs)
            end_idx_unlabeled = start_idx_unlabeled + (self.batch_size - self.labeled_bs)
            batch_from_second_part = second_part[start_idx_unlabeled:end_idx_unlabeled]

            # 将两部分合并为一个批次
            batch_indices.append(batch_from_first_part + batch_from_second_part)

        return iter(batch_indices)

    def __len__(self):
        # 计算批次数量
        num_labeled_batch = (self.labeled_num // (self.labeled_bs))
        num_unlabeled_batch = ((len(self.sampler)-self.labeled_num) // (self.batch_size-self.labeled_bs))
        num_batches = max(num_labeled_batch,num_unlabeled_batch)
        if not self.drop_last and len(self.sampler) % self.batch_size != 0:
            num_batches -= 1
        return num_batches
