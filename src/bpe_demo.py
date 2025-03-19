from transformers import GPT2Tokenizer
import re
from collections import defaultdict

def get_stats(vocab):
    """统计所有相邻字符对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """将词汇表中的字符对合并"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def demonstrate_bpe():
    # 原始文本
    text = "the quick brown fox jumps over the lazy dog og og"
    print("原始文本:", text)
    
    # 步骤1：初始化词汇表
    # 将每个单词拆分成字符序列
    words = text.split()
    vocab = {}
    for word in words:
        # 添加词尾标记</w>
        chars = ' '.join(list(word)) + ' </w>'
        if chars not in vocab:
            vocab[chars] = 1
        else:
            vocab[chars] += 1
    
    print("\n步骤1 - 初始词汇表:")
    for word, freq in vocab.items():
        print(f"'{word}': {freq}")
    
    # 步骤2：迭代合并最频繁的字符对
    num_merges = 20  # 增加合并步骤数量
    print(f"\n开始进行{num_merges}次合并步骤...")
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            print(f"\n提前结束：在第{i+1}步时没有找到可以合并的字符对")
            break
            
        # 找到最频繁的字符对
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        
        print(f"\n步骤2.{i+1} - 合并最频繁的对: {best}")
        print("当前词汇表:")
        for word, freq in vocab.items():
            print(f"'{word}': {freq}")
    
    if i == num_merges - 1:
        print(f"\n完成所有{num_merges}次合并步骤")

    # 步骤3：展示最终的子词单元
    print("\n步骤3 - 最终的子词单元:")
    subwords = set()
    for word in vocab.keys():
        subwords.update(word.split())
    print(sorted(list(subwords)))

    # 对比：使用GPT2的BPE分词器
    print("\n对比：GPT2的BPE分词结果")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    print("GPT2分词结果:", tokens)

if __name__ == "__main__":
    demonstrate_bpe() 