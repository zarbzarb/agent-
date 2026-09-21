# -*- coding: utf-8 -*-
"""
六、实战：使用 Word2vec 做歌曲推荐 —— 第 9 题
=============================================
【题目】下面代码加载播放列表数据集并训练 Word2vec 模型：
        model = Word2Vec(playlists, vector_size=32, window=20, negative=50,
                         min_count=1, workers=4)
（1）在这个任务中，"单词"和"句子"分别对应什么？
（2）解释训练参数 vector_size=32、window=20、negative=50 的含义。

【答案】
（1）这是经典的 **item2vec** 思路 —— 把"物品序列"当成"句子"：
        | 自然语言     | 本任务                       |
        | 单词 word    | 一首歌 song_id               |
        | 句子 sentence| 一条播放列表 playlist        |
        | 语料 corpus  | 全部用户的播放列表集合       |
    于是"经常出现在同一个歌单里的歌"等价于"经常出现在同一句话里的词"，向量会被拉近，
    得到"相似歌曲"。整个过程是纯共现统计的**协同过滤**，无需任何人工标注。
    （注意：歌单内歌曲没有严格顺序，"上下文"实际是"集合内共现"，这也是 window 要设大的原因。）
（2）· vector_size=32 —— 每个歌曲向量的维度是 32。维度越高信息容量越大，但更易过拟合、
      训练更慢；本数据集共现信号相对简单，32 维性价比高。
    · window=20 —— 上下文窗口**半径 20**，即预测某首歌时把它在歌单中前后各 20 首都当作
      上下文。歌单无序，窗口开大 ≈ 覆盖整条歌单，让同歌单内任意两首都能成为正样本对。
    · negative=50 —— **负采样**抽 50 个不相关的歌作负例，替代昂贵的全词表 softmax。
      负例越多梯度估计越准、区分度越强，但每步更慢。
    · min_count=1（附带）—— 出现 ≥1 次的歌都保留。音乐数据长尾严重，默认的 5 会丢掉大量冷门歌。
    · workers=4（附带）—— 4 线程并行训练，模型异步更新，接近线性加速。

【说明】数据集来自作业给出的地址，本地 data/ 已下载好，不存在时才联网下载。
"""

import os
import time

import pandas as pd
from gensim.models import Word2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "song2vec.model")

TRAIN_URL = "https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt"
SONGS_URL = "https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt"


def load_playlists():
    """加载播放列表数据集与歌曲元数据（图 9-1 的代码）。"""
    train_path = os.path.join(DATA_DIR, "train.txt")
    songs_path = os.path.join(DATA_DIR, "song_hash.txt")

    if not (os.path.exists(train_path) and os.path.exists(songs_path)):
        import requests
        os.makedirs(DATA_DIR, exist_ok=True)
        print("本地没有数据集，正在下载…")
        for url, path in [(TRAIN_URL, train_path), (SONGS_URL, songs_path)]:
            open(path, "w", encoding="utf-8").write(requests.get(url).text)

    # train.txt 前两行是元信息，跳过
    lines = open(train_path, encoding="utf-8").read().split("\n")[2:]
    playlists = [s.rstrip().split() for s in lines if len(s) > 1]

    # song_hash.txt 第 1 行是表头，跳过；字段用 tab 分隔
    lines = open(songs_path, encoding="utf-8").read().split("\n")[1:]
    songs = [s.rstrip().split("\t") for s in lines if len(s.split("\t")) >= 3]
    songs_df = pd.DataFrame(songs, columns=["id", "title", "artist"])
    songs_df["id"] = songs_df["id"].astype(str).str.strip()   # 原始文件 id 列后带空格
    songs_df = songs_df.set_index("id")                       # 便于按歌曲 ID 查歌名
    return playlists, songs_df


def main():
    playlists, songs_df = load_playlists()
    print(f"playlists 条数 = {len(playlists):,}")
    print(f"歌曲总数       = {len(songs_df):,}")
    print(f"示例歌单前 5 首 = {playlists[0][:5]}\n")

    # ---- 图 9-2 的代码：训练 Word2vec 歌曲嵌入模型 ----
    t0 = time.time()
    model = Word2Vec(
        playlists,
        vector_size=32,     # 每首歌 → 32 维向量
        window=20,          # 上下文窗口半径 20（几乎覆盖整条歌单）
        negative=50,        # 负采样 50 个负例
        min_count=1,        # 出现 1 次以上都保留
        workers=4,          # 4 线程并行
    )
    print(f"训练完成，耗时 {time.time() - t0:.1f} 秒，词表大小 = {len(model.wv.index_to_key):,}")

    model.save(MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}（供第 10 题使用）")

    # 看一眼学到的向量
    vec = model.wv[playlists[0][0]]
    print(f"歌曲 {playlists[0][0]} 的向量：shape={vec.shape}，前 8 维 = {vec[:8].round(3).tolist()}")

    # gensim 的 C 扩展退出时会往 stderr 打一堆无害的析构噪声，直接结束进程让输出干净
    os._exit(0)


if __name__ == "__main__":
    main()
