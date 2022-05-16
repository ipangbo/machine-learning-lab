import os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

PATH_ROOT = os.getcwd()
PATH_TRAIN = os.path.join(PATH_ROOT, 'lab1\\train.csv')
PATH_TEST = os.path.join(PATH_ROOT, 'lab1\\test.csv')

print("Use train data:", PATH_TRAIN)
print("Use test  data:", PATH_TRAIN)

train_data = pd.read_csv(PATH_TRAIN)
# print(train_data)