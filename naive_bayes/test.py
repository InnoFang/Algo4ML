from naive_bayes import bayes
from numpy import *

list_of_posts, list_classes = bayes.loadDataSet()
my_vocab_list = bayes.createVocabList(list_of_posts)
print(my_vocab_list)

# print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[0]))
# print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[3]))

train_mat = []
for post_in_doc in list_of_posts:
    train_mat.append(bayes.setOfWords2Vec(my_vocab_list, post_in_doc))

p0_v, p1_v, p_ab = bayes.trainNB0(train_mat, list_classes)

print(p0_v)
print(p1_v)
print(p_ab)

bayes.testingNB()
