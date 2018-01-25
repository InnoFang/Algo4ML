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


def textParse(bigString):
    """
    切分文本，使用正则表达式，将文本中的字母和数字分隔出来
    并且为了保证所有词的形式是统一的，所以需要将所有字母都转化为小写
    因为有可能分割出空字符串，还有诸如 en, py 这样的单词，我们想要去掉这些词，所以我们需要过滤掉长度小于3的字符串
    :param bigString: 要切分的文本 
    :return: 词向量
    """
    import re
    list_of_tokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spamText():
    """
    读取数据(25个垃圾邮件，25个非垃圾邮件)，并初始化好文本列表，类别列表，和用于构建词向量的文本向量
    再从中构建一个测试集和一个训练集，两个集合中的邮件都是随机出来的
    整个数据列表有 50 个，从中随机选择 10 个文件做测试集，同时从测试集中剔除
    这种随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程称为 留存交叉验证(hold-out cross validation)
    :return: 
    """
    # 文本列表，保存所有读取到的词向量
    doc_list = []

    # 类别列表，保存每篇文本的类别
    class_list = []

    # 保存所有文本的词向量，并且这个是一维的
    full_text = []

    # 对测试文本数据进行读取，处理，存储
    for i in range(1, 26):
        word_list = textParse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = textParse(open('email/ham/%d.txt' % i, ).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    # 获取测试文本中的词汇列表
    vocab_list = createVocabList(doc_list)

    # 训练集，保存的是文本集的下标，后面会剔除测试集的个数
    training_set = list(range(50))

    # 存储随机选取的测试集，保存的是文本集的下标
    test_set = []
    for i in range(10):
        # 获取随机下标
        rand_index = int(random.uniform(0, len(training_set)))

        # 添加随机选取的测试集
        test_set.append(training_set[rand_index])

        # 将以选取的测试集从训练集中剔除
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        # 添加训练文档对应词集
        train_mat.append(setOfWords2Vec(vocab_list, doc_list[doc_index]))

        # 添加训练文档对应的类别
        train_classes.append(class_list[doc_index])

    # 得到非垃圾邮件的概率，垃圾邮件的概率，以及所有文章中垃圾邮件占比
    p0_v, p1_v, p_spam = trainNB0(array(train_mat), array(train_classes))

    # 记录错误率
    error_count = 0
    for doc_index in test_set:
        word_vector = setOfWords2Vec(vocab_list, doc_list[doc_index])
        # 比对测试结果
        if classifyNB(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1

    # 输出错误率
    print('the error rate is: ', float(error_count) / len(test_set))


def calcMostFreq(vocabList, fullText):
    import operator
    freq_dict = {}
    for token in vocabList:
        freq_dict[token] = fullText.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq


def localWords(feed1, feed0):
    import feedparser
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = textParse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = textParse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    top30_words = calcMostFreq(vocab_list, full_text)
    for pairW in top30_words:
        if pairW[0] in vocab_list:
            vocab_list.remove(pairW[0])
    training_set = list(range(2 * min_len))
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bagOfWords2VecMN(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = trainNB0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = bagOfWords2VecMN(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is: ', float(error_count) / len(test_set))
    return vocab_list, p0_v, p1_v


def getTopWords(ny, sf):
    import operator
    vocab_list, p0_v, p1_v = localWords(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0_v)):
        if p1_v > -6.0: top_sf.append((vocab_list[i], p0_v[i]))
        if p1_v > -6.0: top_ny.append((vocab_list, p1_v[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sorted_sf:
        print(item[0])
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sorted_ny:
        print(item[0])
