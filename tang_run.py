"""
项目功能：输入关键字，生成藏头诗。
项目配置：
补字、限定每行字数、用的新数据集、有建模代码
"""


# 禁用词，包含如下字符的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']

# 最小词频
MIN_WORD_FREQUENCY = 8

# 训练的batch size
BATCH_SIZE = 1

# 数据集路径
DATASET_PATH = './唐诗——简.txt'

# 共训练多少个epoch
TRAIN_EPOCHS = 5

# 最佳权重保存路径
BEST_MODEL_PATH = './best_model_tang.h5'

"""
构建数据集
"""


from collections import Counter
import math
import numpy as np
import tensorflow as tf
import re



class Tokenizer:
    """
    分词器
    """

    def __init__(self, token_dict):
        # 词->编号的映射
        self.token_dict = token_dict
        # 编号->词的映射
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()}
        # 词汇表大小
        self.vocab_size = len(self.token_dict)

    def id_to_token(self, token_id):
        """
        给定一个编号，查找词汇表中对应的词
        :param token_id: 带查找词的编号
        :return: 编号对应的词
        """
        return self.token_dict_rev[token_id]

    def token_to_id(self, token):
        """
        给定一个词，查找它在词汇表中的编号
        未找到则返回低频词[UNK]的编号
        :param token: 带查找编号的词
        :return: 词的编号
        """
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def encode(self, tokens):
        """
        给定一个字符串s，在头尾分别加上标记开始和结束的特殊字符，并将它转成对应的编号序列
        :param tokens: 待编码字符串
        :return: 编号序列
        """
        # 加上开始标记
        token_ids = [self.token_to_id('[CLS]'), ]
        # 加入字符串编号序列
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        # 加上结束标记
        token_ids.append(self.token_to_id('[SEP]'))
        return token_ids

    def decode(self, token_ids):
        """
        给定一个编号序列，将它解码成字符串
        :param token_ids: 待解码的编号序列
        :return: 解码出的字符串
        """
        # 起止标记字符特殊处理
        spec_tokens = {'[CLS]', '[SEP]'}
        # 保存解码出的字符的list
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in spec_tokens:
                continue
            tokens.append(token)
        # 拼接字符串
        return ''.join(tokens)


# 禁用词
disallowed_words = DISALLOWED_WORDS
# 最小词频
min_word_frequency = MIN_WORD_FREQUENCY
# mini batch 大小
batch_size = BATCH_SIZE

# 加载数据集
f = open(DATASET_PATH, 'r', encoding='utf-8')
lines = f.readlines()
poetry = []
for line in lines:
    ignore_flag = False
    for dis_word in disallowed_words:
        if dis_word in line:
            ignore_flag = True
            break
    if ignore_flag:
        continue
    poetry.append(line.replace('\n', ''))

# 统计词频
counter = Counter()
for line in poetry:
    counter.update(line)
# 过滤掉低频词
_tokens = [(token, count) for token, count in counter.items() if count >= min_word_frequency]
# 按词频排序
_tokens = sorted(_tokens, key=lambda x: -x[1])
# 去掉词频，只保留词列表
_tokens = [token for token, count in _tokens]

# 将特殊词和数据集中的词拼接起来
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens
# 创建词典 token->id映射关系
token_id_dict = dict(zip(_tokens, range(len(_tokens))))
# 使用新词典重新建立分词器
tokenizer = Tokenizer(token_id_dict)
# 混洗数据
np.random.shuffle(poetry)

def generate_acrostic(tokenizer, model, head):
    """
    随机生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    """
    # 使用空串初始化token_ids，加入[CLS]
    token_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    # 标点符号，这里简单的只把逗号和句号作为标点
    punctuations = ['，', '。']
    punctuation_ids = {tokenizer.token_to_id(token) for token in punctuations}
    # 缓存生成的诗的listA
    poetry = []
    # 对于藏头诗中的每一个字，都生成一个短句


    for ch in head:
        # 先记录下这个字
        poetry.append(ch)

        # 将藏头诗的字符转成token id
        token_id = tokenizer.token_to_id(ch)

        # 加入到列表中去
        token_ids.append(token_id)

        # 开始生成一个短句
        while True:
            # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
            output = model(np.array([token_ids, ], dtype=np.int32))
            _probas = output.numpy()[0, -1, 3:]
            del output

            # 按照出现概率，对所有token倒序排列
            p_args = _probas.argsort()[::-1][:100]

            # 排列后的概率顺序
            p = _probas[p_args]

            # 先对概率归一
            p = p / sum(p)

            # 再按照预测出的概率，随机选择一个词作为预测结果
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3

            # 保存
            token_ids.append(target)

            # 只有不是特殊字符时，才保存到poetry里面去
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuation_ids:
                break

    return ''.join(poetry)

"""
输入关键字，生成藏头诗
"""

# 加载训练好的模型
model = tf.keras.models.load_model(BEST_MODEL_PATH)

if __name__ == '__main__':
    # 生成藏头诗
    keyword = input('输入关键字:')
    SHOW_NUM = int(input("输出几首诗:"))
    num = int(input("每行多少字:"))

    i = 0
    flag = 0 #判断字数是否足够,0足够
    len_key = len(keyword)
    len_keys = len_key

    if(len_key < 4):
        flag = 1

    keywords = ""

    i = 0
    j = 0
    poem = []

    while i < SHOW_NUM :
        if(flag == 0):
            poem = generate_acrostic(tokenizer, model, head=keyword)
        while flag == 1:
            if(j == 0): #第一次运行
                poem = generate_acrostic(tokenizer, model, head=keyword)
            s = poem[-2] # 截取前一句最后一句
            # print("s:",s)

            if(j == 0): #第一次运行
                keywords = keyword + s
            else:keywords = keywords + s
            # print("keywords:",keywords)

            poem = poem + generate_acrostic(tokenizer, model, head=s)
            len_keys = len_keys + 1
            # print("poem:",poem)
            # print("len_keys:",len_keys)

            if(len_keys < 4):flag = 1
            else:flag = 0
            # print("flag:",flag)

            j = j + 1

        j = 0
        i = i + 1

        str = re.split('，|。', poem) #对诗歌结果进行划
        len_poem = len(str) #有多少行诗
        flag_result = 1 #每行字数符合要求，可输出

        k = 0

        while (k < len_poem - 1) and (flag_result == 1):
            if(len(str[k]) == num):
                #每行诗的字数符合要求
                k = k + 1
            else:   #字数不符合要求
                flag_result = 0

        if(flag_result == 1):
            print(poem)
            # print("result:", poem)
            poem = ""
            keywords = 0
            j = 0
            len_keys = len_key
            if (len_key < 4):   flag = 1
        else:
            i = i - 1
            poem = ""
            keywords = 0
            j = 0
            len_keys = len_key
            if (len_key < 4):   flag = 1