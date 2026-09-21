# -*- coding: utf-8 -*-
"""
五、预训练词嵌入（Word2vec、GloVe） —— 第 8 题
==============================================
【题目】下面代码下载了 GloVe 词向量并查询与 king 最相似的词：
        glove_vectors = api.load("glove-wiki-gigaword-50")
        glove_vectors.most_similar("king")
（1）结果列表中的元组 (queen, 0.7839...) 的两个元素分别是什么含义？
（2）为什么 queen、prince、emperor 会与 king 相似？这体现了词嵌入的什么性质？

【答案】
（1）'queen'  —— 与查询词 king 最**相似**的那个**词**；
    0.7839... —— 两者之间的**余弦相似度（cosine similarity）**，衡量两个向量方向的
                 接近程度，取值范围约 [-1, 1]：越接近 1 越相似，接近 0 表示无关，
                 为负表示语义相反。
    列表第一项是 ('king', 1.0)，因为查询词与自己的相似度为 1
    （不完全等于 1 是浮点误差；也说明 most_similar 会把查询词自己算进去）。
（2）GloVe 在**维基百科级别的超大规模语料**上训练，核心是用"词共现统计"学习向量：
    如果两个词总出现在相似的上下文里（king、queen 周围总跟着 crown / throne / royal /
    palace），它们的向量就会被拉近。king / prince / queen / emperor / ruler / kingdom /
    throne 共享"**君主、王室、统治、权力**"这一整套上下文，所以聚成一簇；
    son / uncle / brother 则因为同属"亲属、血缘、继承"语境被拉了过来。
    这体现了词嵌入的两个核心性质：
    ① **分布式假设（Distributional Hypothesis）/ 语义相似性** —— 一个词的意义由它出现的
       上下文决定，上下文相似的词向量也相似；
    ② **向量的线性可加性** —— king − man + woman ≈ queen，说明"性别""王室"等语义被
       编码成向量空间里稳定的**方向**，因此可以做类比推理。

【说明】作业原代码用 api.load() 从 GitHub 下载（约 66MB，国内很慢）。
        这里改成：本地 data/ 下已有就直接读，没有才走 api.load() 下载。
"""

import os

import gensim.downloader as api
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_TXT = os.path.join(HERE, "data", "glove-wiki-gigaword-50.txt")

if os.path.exists(LOCAL_TXT):
    print(f"从本地文件加载：{LOCAL_TXT}")
    # 该文件首行是 "400000 50" 表头，所以 no_header=False
    glove_vectors = KeyedVectors.load_word2vec_format(LOCAL_TXT, binary=False, no_header=False)
else:
    print("本地没有词向量文件，走作业原代码下载（约 66MB，国内较慢）…")
    glove_vectors = api.load("glove-wiki-gigaword-50")

print(f"词表大小 = {len(glove_vectors.index_to_key):,}，向量维度 = {glove_vectors.vector_size}\n")

# ---- 题目要求：查询与 king 最相似的词 ----
print('glove_vectors.most_similar("king") =')
for word, score in glove_vectors.most_similar("king"):
    print(f"  ('{word}', {score})")

# ---- 附加：向量类比 king - man + woman ≈ queen ----
print('\n向量类比 most_similar(positive=["king", "woman"], negative=["man"]) =')
for word, score in glove_vectors.most_similar(positive=["king", "woman"], negative=["man"], topn=3):
    print(f"  ('{word}', {score:.4f})")
