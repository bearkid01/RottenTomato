from __future__ import division
from collections import Counter
import random 
import os 
import re 
import itertools 
import math 
import sys 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# currently no underflow prevention 
# training dataset and a testing dataset
# function to loop over training dataset and hoepfully do the following things
# function returns word count in each phrase
# function returns a dictionary: key is the class, values is also a dictionary (key as words, values is wordcount)
# return the count of total words in each class 
# function returns a dictionary with words as keys and P(w/c) for each class
# function looks up values in prob_count() function and assign probabilities to unknown words


# randomly select the phrases 
# a dictionary with class as a key and value is another dictionary with keys like phrase id and content 

os.chdir('/Users/bearkid/Downloads')

stopWords = set(['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
       'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
       'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
       'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
       'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
       'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
       'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
       'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
       'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
       'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
       'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
       've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
       'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
       'you', 'your',''])

negWords = 'never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|n\'t'

stopSymb = '[.|:|;|!|?|...]' 

def split_tab(list_of_phrase):
  return [i.split('\t') for i in list_of_phrase]

def read_in(filepath,filename):
  open_file = open(filepath+filename,'r')
  data = open_file.read()
  # break it in a list 
  list_phrase = data.split('\n')
  # remove the header
  dataset = list_phrase[1:]
  return split_tab(dataset)

def split_str(string):
  # remove anything that is not alpha numeric or underscore 
  # first we need to get part of the string that is 
  # regular expression
  string=string.lower()
  string=string+' .'
  search = re.search(negWords,string)
  if search is not None:
    #print string
    ind = string.index(search.group())
    substring = string[ind+len(search.group()):]
    search_symb = re.search(stopSymb,substring)
    if search_symb is not None:
      ind_s = substring.index(search_symb.group())
      subsubstring = substring[:ind_s]
      subsubstring = re.sub(' ', '_neg ',subsubstring[1:])
      subsubstring = ' '+subsubstring
      string = re.sub(string[ind+len(search.group()):ind+len(search.group())+ind_s],subsubstring,string)
      #print string
    else:
      pass
  remove_nonalpha = re.sub("[^(\w|\')]+"," ",string)
  remove_symb = re.sub(stopSymb," ",remove_nonalpha)
  # remove spaces/tabs/...
  remove_space = re.sub("[\s]+"," ",remove_symb)
  #print remove_space
  # split by space 
  list_words = remove_space.split(' ')
  # remove stop words and convert to lower case 
  return [i for i in list_words if i not in stopWords]

def turn_list_into_dict(list_of_phrase):
  big_dict = {}
  for i in list_of_phrase:
    small_dict = {}
    # some missing sentence_id
    try:
      small_dict["sentence_id"]=i[1]
      small_dict["sentiment"]=i[3]
    except IndexError:
      continue
    small_dict["phrase"]=split_str(i[2])
    phrase_id=i[0]
    big_dict[phrase_id]=small_dict
  return big_dict 

'''
def turn_list_into_dict_t(list_of_phrase):
  big_dict = {}
  for i in list_of_phrase:
    small_dict = {}
    # some missing sentence_id
    try:
      small_dict["sentence_id"]=i[1]
    except IndexError:
      continue 
    small_dict["phrase"]=split_str(i[2])
    phrase_id=i[0]
    big_dict[phrase_id]=small_dict
  return big_dict 
'''

def get_nrow(list_of_phrase):
  return len(list_of_phrase)

def generate_random(num, percent):
  return random.sample(range(1,num+1),int(percent*num))

# training_blend doesn't have the structure of each individual phrase 

def build_training(dictionary,select_ID):
  # create an empty list to append dictionaries 
  training_list={}
  training_blend={}
  for i in select_ID:
    if dictionary.has_key(i):
      value_i = dictionary[i]
      phrase = value_i["phrase"]
      sentiment = value_i["sentiment"]
      value_i["phrase_id"] = i 
      if sentiment not in training_list:
        training_list[sentiment] = []
        training_blend[sentiment] = []
        training_list[sentiment].append(value_i)
        training_blend[sentiment].append(phrase)
      else:
        training_list[sentiment].append(value_i)
        training_blend[sentiment].append(phrase)
  return training_list, training_blend


def build_testing(dictionary,select_ID):
  # create an empty list to append dictionaries 
  testing_list=[]
  for i in select_ID:
    if dictionary.has_key(i):
      testing_list.append(dictionary[i])
  return testing_list 


# the purpose of the function is to make a flat list 
# and then apply the Counter function 
def flat_training_blend(training_blend):
  for i in training_blend:
    values = training_blend[i]
    training_blend[i] = Counter(list(itertools.chain(*values)))
  return training_blend

# function returns the total count of words from all the classes combined
def count_unique_words(flat_training_set):
  list_of_keys = []
  for i in flat_training_set:
    list_of_keys.append(flat_training_set[i].keys())
  return set(itertools.chain(*list_of_keys))

def count_words_class(flat_training_set):
  class_words_count = {}
  for i in flat_training_set:
    count = reduce(lambda x,y:x+y,flat_training_set[i].values())
    class_words_count[i] = count
  return class_words_count 
'''

def prob_count(flat_training_blend, unique_words, count_words_each_class):
  for i in flat_training_blend:
    for k in flat_training_blend[i]:  
      flat_training_blend[i][k] = ((flat_training_blend[i][k])+1)/(unique_words+count_words_each_class[i]+1)
  return flat_training_blend


# for each word in each phrase in the testing dataset, lookup the probability if the word
# is in the prob_count dictionary if not assign it with a small probability 
def lookup_probability(word, prob_count_dict, class_type, unique_words, count_words_each_class):
  if prob_count_dict.has_key(class_type):
    # what if the class in your testing data never show up in your training set, very extreme case 
    if prob_count_dict[class_type].has_key(word):
      return prob_count_dict[class_type][word]
    else:
      return 1/(unique_words+count_words_each_class[class_type]+1)
  else:
    pass 
'''

# how to count how many documents have that that words 

# This is too time-consuming 
'''
def count_doc_have_word(bag_of_words_class,training_set_class):

  dictionary_class = {}

  for word in bag_of_words_class:
    count = 0 
    for i in training_set_class:
      if word in set(i['phrase']):
        count = count + 1 
    dictionary_class[word] = count 
  return dictionary_class   
'''

# This is supposed to make your life a little easier 
def count_doc_have_word_premium(training_set_class):
  big_list = []
  for i in training_set_class:
    i['phrase'] = Counter(i['phrase'])
    big_list.append(i['phrase'].keys())

  return Counter(itertools.chain(*big_list))

def count_doc_class(training_set_class):
  return len(training_set_class)

# calculate the percentage of document contain this word 
def prob_doc_class(class_premium, class_doc):
  for i in class_premium:
    class_premium[i] = (class_premium[i]+1)/(class_doc+2)
  return class_premium

def lookup_probability_premium(word,prob_doc_class,count_doc_class):
  if prob_doc_class.has_key(word):
    return prob_doc_class[word]
  else:
    return 1/(count_doc_class+2)

  # what if I can extract all the keys 

def build_training_bag_of_words(training_dict):
  labels = []
  list_phrases = []
  for i in training_dict:
    for j in training_dict[i]:
      list_phrases.append(" ".join(j["phrase"]))
      labels.append(i)

  return pd.Series(labels), pd.Series(list_phrases)


# function returns a dictionary keys are classes and values are unique words 

# how many documents that have this word 

if __name__=="__main__":
  list_read = read_in('/Users/bearkid/Downloads/','train.tsv')
  list_test = read_in('/Users/bearkid/Downloads/','test.tsv')
  #print list_test 
  nrow = get_nrow(list_read)
  dict_test = turn_list_into_dict(list_read)

  nrow_test = get_nrow(list_test)

  #dict_test_t = turn_list_into_dict_t(list_test)

  # select id for training and turn them in str 
  select_id_training = [str(j) for j in generate_random(nrow, 1)]
  #print select_id_training[:10]
  total_id_testing = list(set([str(i) for i in range(1,nrow+1)])-set(select_id_training))

  #print total_id_testing

  #print total_id[:10]
  
  training_set, training_set_blend = build_training(dict_test,select_id_training)
  #training_set_blend is a dictionary with class as keys and an Counter dictionary of words and frequency in each class 
  # the remaining id goes for testing turn them in str 

  testing_set = build_testing(dict_test,total_id_testing)
  
  bag_of_words = flat_training_blend(training_set_blend)

  unique_words = count_unique_words(bag_of_words)
  
  training_labels,training_bag_of_words = build_training_bag_of_words(training_set)

  #print training_bag_of_words.ix[:5]
  #shuffle
  #print training_labels.shape[0]
  arr = np.arange(training_labels.shape[0])
  np.random.shuffle(arr)
  #print arr

  training_labels_shuffled = training_labels.ix[arr]
  training_bag_of_words_shuffled = training_bag_of_words.ix[arr]

  #print training_bag_of_words_shuffled[:5]

  # convert them back to list 
  X = training_bag_of_words_shuffled.tolist()
  y = training_labels_shuffled.tolist()
  y = [int(i) for i in y]

  y = np.array(y)

  kfolds = cross_validation.KFold(len(y), 10)

  i=0

  vectorizer = CountVectorizer(min_df=2) #we only use binary features

  X = vectorizer.fit_transform(X)


  #print X 

  for train_idx, test_idx in kfolds:

    print train_idx, test_idx
    print 'fold', i,
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print X_train

    break 

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print 'score:', score

    i += 1
   




  




  

  




  
















