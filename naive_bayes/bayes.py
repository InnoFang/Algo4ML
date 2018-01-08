from numpy import *

def loadDataSet():
    """
    词表到向量的转换函数
    :return: 
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 侮辱类：1，非侮辱类：0
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复的词汇表
    :param dataSet: 
    :return: 
    """
    vocab_set = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def setOfWords2Vec(vocabList, inputSet):
    """
    
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量
    """
    return_vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            return_vec[vocabList.index(word)] = 1
        else:
            print('The word: %s is not in my Vocabulary!' % word)
    return return_vec


def trainNB0(trainMatrix, trainCategory):
    # 获取文档数目
    num_train_docs = len(trainMatrix)
    # 计算第一篇文档的单词数
    num_words = len(trainMatrix[0])
    # 将该词条的总数除以总词条数得到条件概率
    p_abusive = sum(trainCategory) / float(num_train_docs)
    p0_num = zeros(num_words)
    p1_num = zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):
        if trainCategory[i] == 1:
            p1_num += trainMatrix[i]
            p1_denom += sum(trainMatrix[i])
        else:
            p0_num += trainMatrix[i]
            p0_denom += sum(trainMatrix[i])
    p1_vect = p1_num/p1_denom
    p0_vect = p0_num/p0_denom
    return p0_vect, p1_vect, p_abusive
