import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset

# 设置随机种子，确保结果可复现
torch.manual_seed(1)
# 设置批处理大小为1，表示每次只处理一张图片
batch_size = 1
# 设置最大工作线程数为0，表示不使用多线程
max_workers = 0

# 定义数据预处理流程
# 1. 将图像转换为张量
# 2. 将像素值从[0,1]转换到[0,255]范围，并转换为uint8类型
# 3. 添加一个维度，使形状变为(1, H, W)
transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8).unsqueeze(0)
])

# 加载MNIST数据集
# root='./data': 数据集将保存在./data目录下
# train=True: 使用训练集
# download=True: 如果数据集不存在则下载
# transform=transform: 应用上面定义的数据预处理
print("正在加载MNIST数据集...")
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 只使用前2张图片进行训练（TODO: 样本数量可能不够）
indices = list(range(0, 2))
dataset = Subset(dataset, indices)
print(f"成功加载{len(dataset)}张图片")

if __name__ == "__main__":
    # 初始化分词器
    print("初始化分词器...")
    tokenizer = mbpe.tokenizer.Tokenizer(batch_size=batch_size, max_workers=max_workers)

    # 设置训练参数
    data_name = 0  # 数据集的索引
    max_shape = (1, 2, 2)  # 最大形状，用于控制分块大小
    dim_index = [2, 3, 4]  # 维度索引，用于指定要处理的维度
    min_freq = 0.01  # 最小频率阈值，用于控制合并条件
    root_min_freq = 0.01  # 根节点最小频率阈值
    min_entrance_freq = 0.01  # 入口最小频率阈值

    # 开始训练分词器
    print("开始训练分词器...")
    print(f"参数设置：max_shape={max_shape}, dim_index={dim_index}")
    tokenizer.train(dataset, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq)
    
    # 打印训练得到的词汇表
    print('训练完成！词汇表：')
    print('vocabulary:', tokenizer.get_vocab())
