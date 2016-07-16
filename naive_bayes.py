from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import gc
import time
import datetime

start_time = time.time()
df_train = pd.read_csv('data/gender_age_train_join.csv', dtype={'device_id':str})
df_test = pd.read_csv('data/gender_age_test_join.csv', dtype={'device_id':str})
print 'reading time=', time.time()-start_time

start_time = time.time()
# numeric phone_brand
df = pd.concat([df_train['phone_brand'], df_test['phone_brand']])
a = pd.factorize(df)
phone_brand_labels = a[1]
df_train['phone_brand'] = a[0][0:df_train.shape[0]]
df_test['phone_brand'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

# numeric device_model
df = pd.concat([df_train['device_model'], df_test['device_model']])
a = pd.factorize(df)
device_model_labels = a[1]
df_train['device_model'] = a[0][0:df_train.shape[0]]
df_test['device_model'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

# numeric group
a = pd.factorize(df_train['group'], sort=True)
df_train['group'] = a[0]
group_labels = a[1]

a=df=None
df_train.drop(['text','phone_brand_en'], axis=1, inplace=True)
df_test.drop(['text','phone_brand_en'], axis=1, inplace=True)
df_train['text_brand'] = df_train['text_brand'].str.lower()
df_test['text_brand'] = df_test['text_brand'].str.lower()
gc.collect()
print 'numericing attributes time=', time.time()-start_time

start_time = time.time()
count_vect = CountVectorizer()
df = pd.concat([df_train['text_brand'], df_test['text_brand']])
X_text_counts = count_vect.fit_transform(df)
X_train_text_count = X_text_counts[0:df_train.shape[0],:]
X_test_text_count = X_text_counts[df_train.shape[0]:df_train.shape[0]+df_test.shape[0],:]
print 'vectorizing time=', time.time()-start_time

start_time = time.time()
clf = MultinomialNB(alpha=0.001, fit_prior=True)
# 12 classes
X = X_train_text_count
y = df_train['group'].values
clf.fit(X, y)

X_test = X_test_text_count
y_pred = clf.predict_proba(X_test)
group_list = list(group_labels.values)
for l in group_list:
    df_test[l] = 0
df_test[group_list] = y_pred
# , 'phone_brand_en', 'device_model_en'
df_test.to_csv('output/text_nb-'+\
        str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
        columns=['device_id']+group_list, index=False)

print 'training, predicting time=', time.time()-start_time