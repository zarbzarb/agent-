# -*- coding: utf-8 -*-
"""
四、句子与文本嵌入 —— 第 7 题
=============================
【题目】下面是使用 SentenceTransformer 将文本转换为嵌入的代码：
（1）执行 vector.shape 的结果为 (768,)，它表示什么？
（2）词元嵌入和句子嵌入（文本嵌入）有什么区别？各适合什么任务？

【答案】
（1）一个**一维**张量、长度 768，表示整句话被编码成了一个 768 维的稠密向量。
    与第 6 题对比就能看出关键差别：词元嵌入是"每个词元一个向量"（三维），
    句子嵌入是"整句一个向量"（一维）。shape 里没有 batch 维，是因为传入的是单个字符串；
    传列表就会得到 (N, 768)。有了这个定长向量，就能用余弦相似度直接比较句子语义，
    或建向量索引做检索。
（2）| | 词元嵌入 Token Embedding | 句子嵌入 Sentence / Text Embedding |
      | 粒度     | 每个词元一个向量 | 整句 / 整段一个向量 |
      | 形状     | (batch, seq_len, hidden) 三维 | (hidden,) 或 (batch, hidden) |
      | 语义单位 | 单个词元及其上下文 | 整句整体语义 |
      | 产生方式 | 模型 embedding 层输出 / 各层隐状态 | 句向量模型 + 池化（mean / max / CLS） |
      | 长度     | 随句子长短变化 | 定长，与句子长短无关 |
    联系：句子嵌入通常就是把词元嵌入**聚合**得到（如 mean-pooling 或取 [CLS] 向量）。

    适合的任务：
    · 词元嵌入 → 需要逐词输出或词级对齐的任务：序列标注（NER、词性标注、分词）、
      掩码语言建模、抽取式问答的答案定位，以及作为下游模型的输入特征。
    · 句子嵌入 → 把整段文本当整体去比较 / 检索 / 聚类的任务：语义检索与向量数据库召回
      （RAG）、语义相似度、去重、文本聚类、句级分类、推荐、跨语言对齐。
    一句话记忆：要"看词"用词元嵌入，要"比句子"用句子嵌入。
"""

import os

from sentence_transformers import SentenceTransformer, util

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "all-mpnet-base-v2")
if not os.path.isdir(MODEL):
    MODEL = "sentence-transformers/all-mpnet-base-v2"

model = SentenceTransformer(MODEL)

# 将文本转换为文本嵌入
vector = model.encode("Best movie ever!")
print("vector.shape =", vector.shape, "  ← 一维、768 维，整句话被编码成一个向量")

# 有了定长句向量，就能直接用余弦相似度比较句子语义
sentences = ["Best movie ever!", "A wonderful film.", "I hate this traffic jam."]
emb = model.encode(sentences)
print(f"\n与 {sentences[0]!r} 的余弦相似度：")
for s, v in zip(sentences, emb):
    print(f"  {util.cos_sim(emb[0], v).item():+.4f}  {s!r}")
