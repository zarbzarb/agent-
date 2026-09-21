# -*- coding: utf-8 -*-
"""
六、实战：使用 Word2vec 做歌曲推荐 —— 第 10 题
==============================================
【题目】训练完成后，用下面的代码查询与歌曲 2172 最相似的歌曲：
        song_id = 2172
        model.wv.most_similar(positive=str(song_id))
若从 Michael Jackson 的 "Billie Jean"（歌曲 ID 为 3822）开始推荐：
        print_recommendations(3822)
请观察推荐结果，说说为什么这个模型能做出合理的歌曲推荐。

【答案】
推荐结果非常合理：推荐的歌都落在"80 年代流行 / 舞曲 / 放克"这个风格圈里，
其中还不乏同一位歌手的作品。原因有四层：
1. **共现即相关**：训练语料是**真实用户的播放列表**。同一用户把 Michael Jackson、Prince、
   Madonna 放进同一个歌单，就说明它们"同类"。Skip-gram/CBOW 把经常同歌单出现的歌曲向量
   互相拉近，等价于在做**协同过滤** —— "喜欢 A 的人也喜欢 B" ⟹ B 就是推荐。
2. **无监督学到隐含属性**：训练时只喂了 song_id，没有任何流派/年代/BPM 标签；
   但"哪些歌经常一起出现"这一统计规律背后编码了**流派（funk-pop / dance-pop）、
   年代（80 年代）、歌手、受众**等信息，所以向量空间自动形成"MJ 舞曲区""摇滚区""古典区"等簇，
   推荐结果在风格上高度一致。
3. **相似度分数佐证簇内紧密**：与 2172（Fade To Black / Metallica）最相似的 5 首相似度都在
   0.9 以上，几乎是同一簇的近邻，说明歌手/风格维度被清晰地编码进了向量空间。
4. **简单、高效、可扩展**：不需要音频特征与人工标签，只要能拿到"用户歌单"这种隐式反馈，
   就能给任意歌曲算出邻居；配合 FAISS / HNSW 可支持上亿曲库的毫秒级召回 ——
   这正是工业界 item2vec / 双塔召回方案的思想源头。

【运行】先运行 q09_word2vec_train.py 训练出 song2vec.model。
"""

import os

import pandas as pd
from gensim.models import Word2Vec

from q09_word2vec_train import load_playlists

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "song2vec.model")


def print_recommendations(model, songs_df, song_id, topn=5):
    """作业里调用的推荐函数：给定歌曲 ID，打印最相似的 topn 首歌曲（含标题与艺术家）。"""
    song_id = str(song_id)
    title, artist = (songs_df.loc[song_id, "title"], songs_df.loc[song_id, "artist"]) \
        if song_id in songs_df.index else ("<未知曲目>", "<未知艺术家>")

    print(f"\n从《{title}》—— {artist}（ID={song_id}）开始推荐：\n")
    print(f"{'ID':<8}{'标题':<36}{'艺术家':<28}相似度")
    print("-" * 88)
    for sid, score in model.wv.most_similar(positive=song_id, topn=topn):
        t = songs_df.loc[sid, "title"] if sid in songs_df.index else "<未知>"
        a = songs_df.loc[sid, "artist"] if sid in songs_df.index else "<未知>"
        print(f"{sid:<8}{t:<36}{a:<28}{score:.4f}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("未找到 song2vec.model，请先运行 q09_word2vec_train.py")

    model = Word2Vec.load(MODEL_PATH)
    _, songs_df = load_playlists()

    # ---- 题目第一段代码：查询与歌曲 2172 最相似的歌曲（作业截图给出相似度最高的 5 首）----
    song_id = 2172
    print(f"model.wv.most_similar(positive='{song_id}', topn=5) =")
    for sid, score in model.wv.most_similar(positive=str(song_id), topn=5):
        t = songs_df.loc[sid, "title"] if sid in songs_df.index else "<未知>"
        a = songs_df.loc[sid, "artist"] if sid in songs_df.index else "<未知>"
        print(f"  ('{sid}', {score:.16f})   →  {t} —— {a}")

    # ---- 题目第二段代码：从 Billie Jean(3822) 开始推荐 ----
    print_recommendations(model, songs_df, 3822)

    os._exit(0)          # 跳过 gensim 退出时的无害析构噪声


if __name__ == "__main__":
    main()
