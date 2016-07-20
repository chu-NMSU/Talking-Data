from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import sys
import time
import datetime
import ConfigParser

def train_model_with_feature(config_name, clf_name, clf, X, X_test, y):
    start_time = time.time()
    print 'training size', X.shape, 'test size', X_test.shape
    clf.fit(X, y)
    print 'training time=', time.time()-start_time
    y_prob = clf.predict_proba(X)
    print 'log loss on training data=', log_loss(y, y_prob)

    y_pred = clf.predict_proba(X_test)
    df_test[group_list] = y_pred
    # , 'phone_brand_en', 'device_model_en'
    df_test.to_csv('output/'+config_name+'-'+clf_name+'-'+\
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['device_id']+group_list, index=False)

def preprocess_data(df_train, df_test):
    print 'preprocessing data'
    start_time = time.time()
    # numeric phone_brand
    df = pd.concat([df_train['phone_brand_en'], df_test['phone_brand_en']])
    a = pd.factorize(df)
    phone_brand_labels = a[1]
    df_train['phone_brand_en'] = a[0][0:df_train.shape[0]]
    df_test['phone_brand_en'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

    # numeric device_model
    df = pd.concat([df_train['device_model_en'], df_test['device_model_en']])
    a = pd.factorize(df)
    device_model_labels = a[1]
    df_train['device_model_en'] = a[0][0:df_train.shape[0]]
    df_test['device_model_en'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

    # numeric group
    a = pd.factorize(df_train['group'], sort=True)
    df_train['group'] = a[0]
    group_labels = a[1]
    global group_list
    group_list = list(group_labels.values)
    for l in group_list:
        df_test[l] = 0

    ### fill NA text here TODO fill NA with text from the same phone brand
    # for i in range(0, df_test.shape[0]):
    #     if df_test.isnull().loc[i,'text']:
    #         pb = df_test.loc[i].phone_brand_en
    #         pb_train = df_train[df_train['phone_brand_en']==pb]
    #         if pb_train.shape[0]!=0:
    #             df_test.loc[i, 'text'] = pb_train.loc[pb_train.index[0], 'text']

    df_train['text'].fillna('missing', inplace=True) #fill NA with missing 
    df_test['text'].fillna('missing', inplace=True)

    ## text feature engineering
    start_time = time.time()
    # df_train.drop(['text','phone_brand_en'], axis=1, inplace=True)
    # df_test.drop(['text','phone_brand_en'], axis=1, inplace=True)
    df_train['text'] = df_train['text'].str.lower()
    df_test['text'] = df_test['text'].str.lower()

    df = pd.concat([df_train['text'], df_test['text']])
    count_vect = CountVectorizer() #word count vectorization
    X_text_counts = count_vect.fit_transform(df)
    X_train_text_count = X_text_counts[0:df_train.shape[0],:]
    X_test_text_count = X_text_counts[df_train.shape[0]:df_train.shape[0]+df_test.shape[0],:]

    tfidf_vect = TfidfVectorizer() #tfidf count vectorization
    X_text_tfidf = tfidf_vect.fit_transform(df)
    X_train_text_tfidf = X_text_tfidf[0:df_train.shape[0],:]
    X_test_text_tfidf = X_text_tfidf[df_train.shape[0]:df_train.shape[0]+df_test.shape[0],:]

    a=df=None
    gc.collect()
    print 'numericing, vectorizing attributes time=', time.time()-start_time
    return df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, X_test_text_count

if __name__=='__main__':
    config_path = sys.argv[1]
    config_name = sys.argv[2]
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels = config.get(config_name, 'labels').split(',')
    clf_name = config.get(config_name, 'clf')

    start_time = time.time()
    global group_list, df_test
    df_train = pd.read_csv('data/train_text.csv', dtype={'device_id':str})
    df_test = pd.read_csv('data/test_text.csv', dtype={'device_id':str})
    print 'reading time=', time.time()-start_time

    df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, \
            X_test_text_count = preprocess_data(df_train, df_test)

    start_time = time.time()
    X = X_test = None
    y = df_train['group'].values
    if 'text_wc' in labels[0]:
        X = X_train_text_count
        X_test = X_test_text_count
    elif 'text_tfidf' in labels[0]:
        X = X_train_text_tfidf
        X_test = X_test_text_tfidf
    else: # other features
        X = df_train[labels].values
        X_test = df_test[labels].values

    clf = None
    if clf_name=='nb':
        # parameters = # TODO
        clf = MultinomialNB(alpha=0.001, fit_prior=True)
    elif clf_name=='lr':
        clf = LogisticRegression()
    elif clf_name=='rf':
        clf = RandomForestClassifier(n_jobs=8)
    elif clf_name=='xgb':
        clf = xgb.XGBClassifier(nthread=8, n_estimators=500, \
                subsample=0.5, colsample_bytree=0.5, colsample_bylevel=0.9)

    train_model_with_feature(config_name, clf_name, clf, X, X_test, y)
