from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List
from .utils import *
from .patch import *
from .frequency_counter import FrequencyCounter
import concurrent.futures
import threading
import torch
from torch import Tensor, dtype
from torch.utils.data import DataLoader

global_count = 0

@dataclass
class State:
    """存储训练过程中的当前状态
    TODO: 应该改为标准类，使用dataclass可能不够灵活
    """
    shape: List[int]  # 当前形状
    tensor: Tensor  # 只保留非根结点的张量
    data_dtype: dtype  # 数据类型
    orig_size: List[int]  # 原始大小
    tuple_indices: List[int]  # 存储所有非根节点的位置
    code_list: List[int]  # 存储所有根节点的编码值
    code_indices: List[int]  # 存储所有根节点的位置
    joined_list: List[int]  # 存储完整的编码序列


class BaseTokenizer:
    """分词器基类"""

    def __init__(self):
        self.vocab = dict()  # 词汇表
        self.inverse_vocab = dict()  # 反向词汇表

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def train(self, data, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq):
        # 训练词汇表的方法，需要被子类实现
        raise NotImplementedError

    def encode(self, data, max_shape):
        # 使用训练好的词汇表进行编码的方法，需要被子类实现
        raise NotImplementedError

    def decode(self, encoded):
        # 将编码后的数据解码回原始数据的方法，需要被子类实现
        raise NotImplementedError


class Tokenizer(BaseTokenizer):
    def __init__(self, batch_size=1, max_workers=None):
        super().__init__()
        self.batch_size = batch_size  # 批处理大小
        self.max_workers = max_workers  # 最大工作线程数
        self._lock = threading.Lock()  # 线程锁，用于保护共享资源

    def train(self, dataset, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq):
        """训练分词器的主要方法
        
        Args:
            dataset: 训练数据集
            data_name: 数据名称/索引
            max_shape: 最大形状
            dim_index: 维度索引
            min_freq: 最小频率阈值
            root_min_freq: 根节点最小频率阈值
            min_entrance_freq: 入口最小频率阈值
        """
        print("开始数据加载...")
        # 创建数据加载器，不进行随机打乱
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # 根据最大形状生成所有可能的形状组合
        # [[1, 2, 2], [1, 1, 2], [1, 1, 1]]
        shapes = find_tuple_shapes(max_shape)
        print('找到的形状组合:{}'.format(shapes))
        
        # 初始化状态字典，用于存储所有图片的状态
        states = {}

        # 对每个形状进行处理
        for shape_idx, shape in enumerate(shapes):
            print(f"\n处理形状 {shape_idx + 1}/{len(shapes)}: {shape}")
            # 创建分块器
            patchify = Patchify(shape, dim_index)
            # 创建频率计数器
            freq_counter = FrequencyCounter(min_entrance_freq, root_min_freq)
            index_tracker = 0

            # 处理每个批次的数据
            for batch_idx, data in enumerate(data_loader):
                print(f"处理批次 {batch_idx + 1}")
                data = data[data_name]
                for i in range(len(data)):
                    # 保持维度不变 [1, 1, 1, 28, 28]
                    data_i = data[[i]]
                    if index_tracker not in states:
                        # 创建新的状态对象
                        state_i = State(
                            shape=shape,
                            tensor=data_i,
                            data_dtype=data_i.dtype,
                            orig_size=list(data_i.size()),
                            tuple_indices=[],
                            code_list=[],
                            code_indices=[],
                            joined_list=[]
                        )
                        states[index_tracker] = state_i
                        print(f"\nPatchify 操作前张量形状: {data_i.shape}")
                        pachified_data = patchify(data_i)[0]
                        print(f"Patchify 操作后张量形状: {pachified_data.shape}")
                        # 更新频率表
                        freq_counter.update_freq_tables(pachified_data, self.vocab, self.inverse_vocab)
                    else:
                        # 更新现有状态
                        state_i = states[index_tracker]
                        state_i.shape = shape
                        pachified_data = patchify(state_i.tensor)[0]
                        freq_counter.update_freq_tables(pachified_data, self.vocab, self.inverse_vocab)
                    
                    # 处理根节点
                    updated_state = self._process_batch_root(state_i, patchify)
                    if updated_state is not None:
                        states[index_tracker] = updated_state

            # 合并频繁出现的对
            print(f"开始合并形状 {shape} 的频繁对...")
            self._merge_pairs(states, min_freq, min_entrance_freq)
        return

    def _process_batch_root(self, state, patchify):
        """处理单个批次的根节点
        
        Args:
            state: 当前状态
            patchify: 分块器对象
            
        Returns:
            更新后的状态对象
        """
        # 计算缩放因子 14
        scale_factor = patchify.get_scale_factor(state.orig_size)
        
        # 如果有已合并的数据，先进行分割
        if len(state.joined_list) > 0:
            state = self._split_data(state, scale_factor)

        if state is not None:
            # 处理根节点状态
            state = self._process_root_state(state, patchify)
            # 将张量转换为元组列表，注意只包含非根结点数据
            tuple_list = tensor_to_tuple(state.tensor, state.shape)[0]
            # 合并回列表用于后续合并
            # 这里将已编码的值，和未编码的元组放在一起
            # 类似这种结构 [tuple(1,2,3,4), "1", tuple(1,2,3,4)]
            state.joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)

        return state

    def _process_root_state(self, state, patchify):
        """处理根节点状态
        
        Args:
            state: 当前状态
            patchify: 分块器对象
            
        Returns:
            更新后的状态对象
        """
        # 对张量进行分块处理 
        unshuffled_tensor = patchify(state.tensor)
        # [1, 196, 1, 2, 2]
        state.orig_size = list(unshuffled_tensor.size())

        # 判断是否是最后一次迭代（所有维度都为1）,也就是最后一种情况 [1, 1, 1]
        last_shape = True if all(dim == 1 for dim in state.shape) else False

        # 同上取pachified_data操作
        tensor = unshuffled_tensor[0]
        # 转元组，方便作为字典的key
        tuple_list = tensor_to_tuple(unshuffled_tensor, state.shape)[0]
        # 创建编码映射，初始值为-1， tensor.size(0) = 196，也就是长度为196的list
        code_mapping = torch.full((tensor.size(0),), -1, dtype=torch.long)

        # print(self.inverse_vocab)
        # raise

        # 使用根词汇表更新编码映射
        with self._lock:
            for i, tup in enumerate(tuple_list):
                code = self.inverse_vocab.get(tup, None)
                # 因为 info['global_frequency'] >= self.min_root_freq 条件限制，所以会有找不到对应code的情况
                if code is not None:
                    code_mapping[i] = int(code)
                else:
                    # 最后一轮才更新词汇表
                    if last_shape:
                        code_mapping[i] = len(self.vocab)
                        update_vocab(self.vocab, self.inverse_vocab, tup, len(self.vocab))

        # 分离根节点和非根节点的索引
        # 根节点是已经确定位置和值的节点
        # 非根节点是还未确定最终位置的节点，即-1的位置
        non_root_indices = torch.where(code_mapping == -1)[0]
        root_indices = torch.where(code_mapping != -1)[0]
        root_codes = code_mapping[root_indices].tolist()

        # 更新状态
        state.tensor = tensor[non_root_indices].unsqueeze(0)
        state.code_list.extend(list(map(str, root_codes)))

        # 检查是否有历史索引记录
        # 如果有，需要将当前索引映射回原始索引
        # 如果没有，直接使用当前索引
        if len(state.tuple_indices) > 0:
            root_indices = torch.tensor(state.tuple_indices)[root_indices].tolist()
            non_root_indices = torch.tensor(state.tuple_indices)[non_root_indices].tolist()
        else:
            root_indices = root_indices.tolist()
            non_root_indices = non_root_indices.tolist()

        # 更新状态
        state.tuple_indices = non_root_indices
        state.code_indices.extend(root_indices)

        # print("tuple_indices", state.tuple_indices)
        # print("code_indices",state.code_indices)
        # print("code_list",state.code_indices)
        # raise

        return state

    def _split_data(self, state, scale_factor):
        """分割数据为元组和编码列表
        
        Args:
            state: 当前状态
            scale_factor: 缩放因子
            
        Returns:
            更新后的状态对象或None
        """
        tuple_list, tuple_indices, code_list, code_indices = split(state.joined_list, scale_factor)


        # print("state.code_list old", state.code_list)
        # print("state.code_list new", code_list)
        # print(state.orig_size)
        # raise
        
        if len(tuple_list) > 0:
            #  orig_size 第一次为 [1, 196, 1, 2, 2] 第二次为 [1, 82, 1, 2, 2] 因为split出来的tuple_list是非根结点
            state.orig_size[1] = len(tuple_list)
            state.tensor = tuple_to_tensor(tuple_list, state.shape, state.orig_size, state.data_dtype)
            state.tuple_indices = tuple_indices
            state.code_list = code_list
            state.code_indices = code_indices
            return state

        return None

    def _merge_pairs(self, states, min_freq, min_entrance_freq):
        """合并频繁出现的对
        
        Args:
            states: 状态字典
            min_freq: 最小频率阈值
            min_entrance_freq: 入口最小频率阈值

        这种迭代设计是 BPE 算法的核心特性：
        1.统计当前所有数据的频率
        2.找出最频繁的对
        3.在所有数据上应用合并
        4.重复这个过程
        """
        while True:
            # 创建新的频率计数器
            counter = FrequencyCounter(min_entrance_freq)
            
            # 更新合并频率表
            for state_i in states.values():
                counter.update_merge_freq_tables(state_i.joined_list)

            # 获取全局频率表
            freq_table = counter.get_global_freq_table()
            if len(freq_table) == 0:
                break

            # 获取频率最高的对
            pair, freq = counter.get_max_merge_pair().values()
            print(f"当前最频繁的对: {pair}, 频率: {freq}")
            
            # 如果频率低于阈值，停止合并
            if freq < min_freq:
                break
                
            # 获取或创建新的编码
            code = self.inverse_vocab.get(pair, None)
            if code is None:
                idx = str(len(self.vocab))
                update_vocab(self.vocab, self.inverse_vocab, pair, idx)
            else:
                idx = code
                
            # 更新所有状态的合并列表
            for batch_states in states.values():
                if isinstance(batch_states, list):
                    for state in batch_states:
                        state.joined_list = merge(state.joined_list, pair, idx)
                else:
                    batch_states.joined_list = merge(batch_states.joined_list, pair, idx)

    def encode(self, data, max_shape):
        """使用训练好的词汇表进行编码
        
        Args:
            data: 输入数据
            max_shape: 最大形状
            
        Returns:
            编码后的元组列表
        """
        if len(self.vocab) == 0:
            raise ValueError('词汇表尚未训练。')

        tuple_list = data
        shapes = find_tuple_shapes(max_shape)

        for i in range(len(shapes)):
            if i > 0:
                tuple_list = tuple_reshape(tuple_list, shapes[i - 1], shapes[i])

            # 使用根词汇表更新
            for i, t in enumerate(tuple_list):
                if t in self.inverse_vocab.keys():
                    tuple_list[i] = self.inverse_vocab[t]

            # 合并频繁对
            while True:
                stats = get_freq_pairs(tuple_list)
                pair, _ = get_max_pair(stats)

                if pair not in self.inverse_vocab.keys():
                    break

                idx = self.inverse_vocab[pair]
                tuple_list = merge(tuple_list, pair, idx)

        return tuple_list

    def decode(self, encoded):
        """将编码后的数据解码回原始数据
        
        Args:
            encoded: 编码后的数据
            
        Returns:
            解码后的原始数据列表
        """
        decoded = []
        for i in encoded:
            pair = self.vocab[i]
            decoded.append(dfs(pair, self.vocab))

        return np.concatenate(decoded).tolist()#
