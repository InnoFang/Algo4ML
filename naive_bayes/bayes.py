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
        # 创建两个集合的并集，即将两个 set 集合合并
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型：将每个词的出现与否作为一个特征，每个词只能出现一次
    将词集转化为数字向量，内容为 0 或 1，1 表示在单词出现在词汇表中，并且位置为词汇表的中该单词的位置
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


def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型，每个单词可以出现多次
    :param vocabList: 
    :param inputSet: 
    :return: 
    """
    return_vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            return_vec[vocabList.index(word)] += 1
    return return_vec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 
    :param trainCategory: 
    :return: 
      p0_vect : 非侮辱类文章中对应单词在所有这类文章中出现的概率
      p1_vect : 侮辱类文章中对应单词在所有这类文章中出现的概率
      p_abusive ：整个训练集中，侮辱类文章出现的概率
    """
    # 获取训练文档数目
    num_train_docs = len(trainMatrix)

    # 获取单词长度，因为每篇训练文章都使用 @setOfWords2Vec 转化为了数字向量，所以此处长度相等
    num_words = len(trainMatrix[0])

    # 因为标签中，1 表示侮辱类，0 表示非侮辱类，所以此处用该标签的和除以文章总数，可以得到在训练集中，侮辱类文章的概率
    p_abusive = sum(trainCategory) / float(num_train_docs)

    # 记录非侮辱类文章每个单词出现的次数
    # 初始化为 1 是为了防止之后算条件概率时因为 0 概率的影响使最终结果变成 0
    p0_num = ones(num_words)

    # 记录侮辱类文章每个单词出现的次数
    p1_num = ones(num_words)

    # 记录非侮辱类文章的单词数总和
    # 初始化为 2 的理由是为了防止出现概率为 1 的情况，所以此处不能初始化为 1
    p0_denom = 2.0

    # 记录侮辱类文章的单词数总和
    p1_denom = 2.0

    # 对每篇文章的类别进行判断并计数，对每篇文章每个单词出现的次数记录，以及每篇文章的单词总数
    for i in range(num_train_docs):
        if trainCategory[i] == 1:
            p1_num += trainMatrix[i]
            p1_denom += sum(trainMatrix[i])
        else:
            p0_num += trainMatrix[i]
            p0_denom += sum(trainMatrix[i])

    # 侮辱类文章中对应单词在所有这类文章中出现的概率
    # 这里取对数的原因是为了防止出现溢出的情况（当多个很小的数相乘，会得到一个更小的数，此时四舍五入会得到 0）
    # 在代数中 ln(a*b) = ln(a) + ln(b)，所以通过对数可以避免下溢出或浮点数舍入导致的错误
    # 同时，f(x) 与 f(ln(x)) 会在相同区域内同时增加或减少，并且会在相同点取得极致。虽然取值不同，但是不影响最终结果
    p1_vect = log(p1_num / p1_denom)

    # 非侮辱类文章中对应单词在所有这类文章中出现的概率
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 元素相乘，因为是对数运算，所以直接加即可
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    list_of_posts, list_classes = loadDataSet()
    my_vocab_list = createVocabList(list_of_posts)
    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.append(setOfWords2Vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = trainNB0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(setOfWords2Vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(setOfWords2Vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0_v, p1_v, p_ab))
