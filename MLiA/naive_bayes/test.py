# list_of_posts, list_classes = bayes.loadDataSet()
# my_vocab_list = bayes.createVocabList(list_of_posts)
# print(my_vocab_list)
#
# # print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[0]))
# # print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[3]))
#
# train_mat = []
# for post_in_doc in list_of_posts:
#     train_mat.append(bayes.setOfWords2Vec(my_vocab_list, post_in_doc))
#
# p0_v, p1_v, p_ab = bayes.trainNB0(train_mat, list_classes)
#
# print(p0_v)
# print(p1_v)
# print(p_ab)
#
# bayes.testingNB()
# # Example 1 spam text
# import re
#
# reg_ex = re.compile('\\W+')
#
# email_text = open('email/ham/6.txt').read()
# list_of_tokens = reg_ex.split(email_text)
# print(list_of_tokens)
# for i in range(10):
#     bayes.spamText()
# Example 2
# 从美国两个城市中选取一些人
# 通过分析这些人发布的征婚广告信息
# 来比较这两个城市的人们在广告用词上是否不同
# （若结论确实不同，那个字常用词是那些？从人们的用词中，我们能否对不同城市的人所关心的内容有所了解）
import feedparser

from MLiA.naive_bayes import bayes

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocab_list, p_sf, p_ny = bayes.localWords(ny, sf)
# vocab_list, p_sf, p_ny = bayes.localWords(ny, sf)
bayes.getTopWords(ny, sf)
