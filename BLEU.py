import jieba
from nltk.translate.bleu_score import sentence_bleu


class BLEU:
    def __init__(self, source, target, inference):
        self.source = source
        self.target = target
        self.inference = inference

    def evaluate(self):
        source_fenci = ' '.join(jieba.cut(self.source))
        target_fenci = ' '.join(jieba.cut(self.target))
        inference_fenci = ' '.join(jieba.cut(self.inference))

        # reference是标准答案 是一个列表，可以有多个参考答案，每个参考答案都是分词后使用split()函数拆分的子列表
        # # 举个reference例子
        # reference = [['this', 'is', 'a', 'duck']]
        reference = []  # 给定标准译文
        candidate = []  # 神经网络生成的句子
        # 计算BLEU
        reference.append(target_fenci.split())
        candidate = (inference_fenci.split())
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
        reference.clear()
        print('Cumulate 1-gram :%f' \
              % score1)
        print('Cumulate 2-gram :%f' \
              % score2)
        print('Cumulate 3-gram :%f' \
              % score3)
        print('Cumulate 4-gram :%f' \
              % score4)
