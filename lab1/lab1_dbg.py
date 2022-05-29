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
test_data = pd.read_csv(PATH_TEST)

categories = train_data['category'].unique()
print(categories)

categories_type = CategoricalDtype(categories = categories)
train_data['category'] = train_data['category'].astype(categories_type).cat.codes.astype('long')
test_data['category'] = test_data['category'].astype(categories_type).cat.codes.astype('long')
# print(train_data['category'].head(), "\n", test_data['category'].head(), sep="")

train_y = train_data['category']
train_x = train_data['review']

test_y = test_data['category']
test_x = test_data['review']

vector = CountVectorizer()
train_x = vector.fit_transform(train_x).toarray()


test_x = vector.transform(test_x).toarray()
# print(test_x)

train_x = np.append(train_data[['latitude','longitude','mean_checkin_time']],train_x,axis=1)

test_x = np.append(test_data[['latitude','longitude','mean_checkin_time']],test_x,axis=1)



nb = GaussianNB()
nb.fit(train_x, train_y)


print(nb.score(test_x,test_y))