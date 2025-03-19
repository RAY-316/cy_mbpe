import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.mbpe.tokenizer import Tokenizer
from torch.utils.data import Subset

def visualize_mbpe_process(image, tokenizer, max_shape):
    """可视化MBPE处理过程"""
    # 获取原始图像
    plt.figure(figsize=(15, 5))
    
    print("原始图像形状:", image.shape)
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    
    # 编码图像
    from pipeline.mbpe.utils import pixel_to_tuples
    tuples = pixel_to_tuples(image, max_shape)
    encoded = tokenizer.encode(tuples, max_shape)
    print("编码后的数据长度:", len(encoded))
    
    # 解码图像
    decoded = tokenizer.decode(encoded)
    print("解码后的数据长度:", len(decoded))
    decoded_image = torch.tensor(decoded).reshape(28, 28)  # 直接使用MNIST图像大小
    
    # 显示解码后的图像
    plt.subplot(1, 3, 2)
    plt.imshow(decoded_image, cmap='gray')
    plt.title('Decoded Image')
    
    # 显示差异
    plt.subplot(1, 3, 3)
    plt.imshow((image.squeeze() - decoded_image).abs(), cmap='gray')
    plt.title('Difference Map')
    
    plt.tight_layout()
    plt.show()
    
    # 打印词汇表大小
    print(f"MBPE词汇表大小: {len(tokenizer.get_vocab())}")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x * 255).to(dtype=torch.uint8)  # 移除unsqueeze(0)
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 使用更多图片进行训练
    dataset = Subset(dataset, list(range(10)))  # 使用10张图片
    
    # 初始化MBPE分词器
    mbpe_tokenizer = Tokenizer()
    
    # 设置MBPE参数
    max_shape = (14, 14)  # 使用更大的块大小
    min_freq = 2  # 提高最小频率阈值
    root_min_freq = 2  # 提高根节点最小频率阈值
    
    # 训练MBPE分词器
    print("训练MBPE分词器...")
    # 使用所有训练图片
    all_tuples = []
    for i in range(10):
        image, _ = dataset[i]
        from pipeline.mbpe.utils import pixel_to_tuples
        tuples = pixel_to_tuples(image, max_shape)
        all_tuples.extend(tuples)
    print(f"总共生成了 {len(all_tuples)} 个元组")
    mbpe_tokenizer.train(all_tuples, max_shape, min_freq, root_min_freq)
    
    # 获取第一张图片并可视化MBPE过程
    print("\nMBPE处理过程:")
    image, _ = dataset[0]
    visualize_mbpe_process(image, mbpe_tokenizer, max_shape)


if __name__ == "__main__":
    main() 