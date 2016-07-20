from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import gc
import sys
import time
import datetime
import ConfigParser

def train_model_with_feature(model_name, clf, X, X_test, y):
    start_time = time.time()
    print 'training size', X.shape, 'test size', X_test.shape
    clf.fit(X, y)
    print 'training time=', time.time()-start_time

    y_pred = nb.predict_proba(X_test)
    df_test[group_list] = y_pred
    # , 'phone_brand_en', 'device_model_en'
    df_test.to_csv('output/'+model_name+'-'+\
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['device_id']+group_list, index=False)

if __name__=='__main__':
    config_path = sys.argv[1]
    config_name = sys.argv[2]
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels = config.get(config_name, 'labels').split(',')

    start_time = time.time()
    df_train = pd.read_csv('data/train_text.csv', dtype={'device_id':str})
    df_test = pd.read_csv('data/test_text.csv', dtype={'device_id':str})
    print 'reading time=', time.time()-start_time

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

    ## text feature engineering
    ### word count does not work well
    start_time = time.time()
    df = pd.concat([df_train['text_brand'], df_test['text_brand']])
    count_vect = CountVectorizer()
    X_text_counts = count_vect.fit_transform(df)
    X_train_text_count = X_text_counts[0:df_train.shape[0],:]
    X_test_text_count = X_text_counts[df_train.shape[0]:df_train.shape[0]+df_test.shape[0],:]

    tfidf_vect = TfidfVectorizer()
    X_text_tfidf = tfidf_vect.fit_transform(df)
    X_train_text_tfidf = X_text_tfidf[0:df_train.shape[0],:]
    X_test_text_tfidf = X_text_tfidf[df_train.shape[0]:df_train.shape[0]+df_test.shape[0],:]

    a=df=None
    # df_train.drop(['text','phone_brand_en'], axis=1, inplace=True)
    # df_test.drop(['text','phone_brand_en'], axis=1, inplace=True)
    df_train['text_brand'] = df_train['text_brand'].str.lower()
    df_test['text_brand'] = df_test['text_brand'].str.lower()
    df_train.fillna('missing', inplace=True) #fill NA with missing. TODO fill NA with group mean
    df_test.fillna('missing', inplace=True)
    gc.collect()
    print 'numericing attributes time=', time.time()-start_time

    start_time = time.time()
    nb = MultinomialNB(alpha=0.1, fit_prior=True)
    rf = RandomForestClassifier(n_jobs=8)
    lr = LogisticRegression(solver='sag')

    print 'vectorizing time=', time.time()-start_time

    global group_list, df_test
    group_list = list(group_labels.values)
    for l in group_list:
        df_test[l] = 0
    
    if 'text' in labels[0]:
        pass
    else:
        ## phone_brand_en, device_model_en feature
        ##################### 
        X = df_train[['phone_brand_en','device_model_en']].values
        X_test = df_test[['phone_brand_en','device_model_en']].values
        y = df_train['group'].values
        print 'training size', X.shape, 'test size', X_test.shape

        nb.fit(X, y)
        print 'training time=', time.time()-start_time

        start_time = time.time()

        y_pred = nb.predict_proba(X_test)
        group_list = list(group_labels.values)
        for l in group_list:
            df_test[l] = 0
        df_test[group_list] = y_pred
        # , 'phone_brand_en', 'device_model_en'
        df_test.to_csv('output/phone_brand_device_model-nb-'+\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['device_id']+group_list, index=False)

        print 'predicting time=', time.time()-start_time
        ##################### 

        ## text word count feature
        ##################### 
        X = X_train_text_count
        X_test = X_test_text_count
        y = df_train['group'].values
        print 'training size', X.shape, 'test size', X_test.shape

        nb.fit(X, y)
        print 'training time=', time.time()-start_time

        start_time = time.time()

        y_pred = nb.predict_proba(X_test)
        group_list = list(group_labels.values)
        for l in group_list:
            df_test[l] = 0
        df_test[group_list] = y_pred
        # , 'phone_brand_en', 'device_model_en'
        df_test.to_csv('output/word_count-nb-'+\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['device_id']+group_list, index=False)

        print 'predicting time=', time.time()-start_time
        ##################### 

        ## text word tfidf feature
        ##################### 
        X = X_train_text_tfidf
        X_test = X_test_text_tfidf
        y = df_train['group'].values
        print 'training size', X.shape, 'test size', X_test.shape

        nb.fit(X, y)
        print 'training time=', time.time()-start_time

        start_time = time.time()

        y_pred = nb.predict_proba(X_test)
        group_list = list(group_labels.values)
        for l in group_list:
            df_test[l] = 0
        df_test[group_list] = y_pred
        # , 'phone_brand_en', 'device_model_en'
        df_test.to_csv('output/word_count-nb-'+\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['device_id']+group_list, index=False)

        print 'predicting time=', time.time()-start_time
        ##################### 
