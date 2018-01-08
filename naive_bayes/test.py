from naive_bayes import bayes

list_of_posts, list_classes = bayes.loadDataSet()
my_vocab_list = bayes.createVocabList(list_of_posts)
print(my_vocab_list)

print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[0]))
print(bayes.setOfWords2Vec(my_vocab_list, list_of_posts[3]))

