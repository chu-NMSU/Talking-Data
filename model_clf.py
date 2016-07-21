from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, train_test_split
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
import logging
import datetime
import ConfigParser

# create logger
logger = logging.getLogger('Talking-Data-model')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

#### "application" code
# logger.debug("debug message")
# logger.info("info message")
# logger.warn("warn message")
# logger.error("error message")
# logger.critical("critical message")

def train_model_with_feature(config_name, clf_name, clf, X, X_test, y):
    logger.info('start training')
    print 'training size', X.shape, 'test size', X_test.shape
    X_train, X_val, y_train, y_val = train_text_China(X, y, train_size=0.9)
    clf.fit(X_train,y_train)
    logger.infor('train log-loss='+str(log_loss(y_train, clf.predict_proba(X_train))))
    logger.infor('validate log-loss='+str(log_loss(y_val, clf.predict_proba(X_val))))


    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    df_test[group_list] = y_pred
    logger.info('finish training')
    # , 'phone_brand_en', 'device_model_en'
    df_test.to_csv('output/'+config_name+'-'+clf_name+'-'+\
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['device_id']+group_list, index=False)
    logger.info('finish outputing result')

def fill_na_test(df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, \
        X_train_text_count, X_test_text_count, option):
    if option=='rand_text':
        # fill NA with first text from the same phone brand. very slow
        for i in range(0, df_test.shape[0]):
            if i%1000==0:
                logger.info('processed #'+str(i)+'/'+str(df_test.shape[0]))
            if len(df_test.loc[i,'text'])==0:
                pb = df_test.loc[i].phone_brand_en
                pb_train = df_train[df_train['phone_brand_en']==pb]
                if pb_train.shape[0]!=0:
                    X_test_text_count[i,:] = X_train_text_count[\
                        pb_train.index[np.random.randint(0,pb_train.shape[0])],:]
    if option=='pb_mean':
        # fill NA with average phone brand vector mean
        pb_group = df_train.groupby('phone_brand_en')
        for i in range(0, df_test.shape[0]):
            if i%1000==0:
                logger.info('processed #'+str(i)+'/'+str(df_test.shape[0]))
            if len(df_test.loc[i,'text'])==0:
                if df_test.loc[i,'phone_brand_en'] in pb_group.groups:
                    group = pb_group.get_group(df_test.loc[i,'phone_brand_en'])
                    X_test_text_count[i,:] = X_train_text_count[group.index,:].mean(axis=0)

    return df_test, X_test_text_tfidf, X_test_text_count

def preprocess_data(df_train, df_test, fill_na_opt):
    logger.info('start preprocessing data')

    ## text feature engineering
    df_train['text'] = df_train['text'].str.lower()
    df_test['text'] = df_test['text'].str.lower()
    df_test['text'] = df_test['text'].fillna('')

    df = pd.concat([df_train['text'], df_test['text']])
    count_vect = CountVectorizer() #word count vectorization
    X_text_counts = count_vect.fit_transform(df).toarray()
    X_train_text_count = X_text_counts[0:df_train.shape[0],:]
    X_test_text_count = X_text_counts[df_train.shape[0]:,:]

    tfidf_vect = TfidfVectorizer() #tfidf count vectorization
    X_text_tfidf = tfidf_vect.fit_transform(df).toarray()
    X_train_text_tfidf = X_text_tfidf[0:df_train.shape[0],:]
    X_test_text_tfidf = X_text_tfidf[df_train.shape[0]:,:]

    # TODO refactor filling strategy later
    df_test, X_test_text_tfidf, X_test_text_count = fill_na_test(df_train, df_test, \
        X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, X_test_text_count, fill_na_opt)

    # vectorizing phone_brand device_model
    df = pd.concat([df_train['phone_brand_en']+' '+df_train['device_model_en'], \
            df_test['phone_brand_en']+' '+df_test['device_model_en']])

    X_pd_counts = count_vect.fit_transform(df).toarray()
    X_train_pd_count = X_pd_counts[0:df_train.shape[0],:]
    X_test_pd_count = X_pd_counts[df_train.shape[0]:,:]

    X_pd_tfidf = tfidf_vect.fit_transform(df).toarray()
    X_train_pd_tfidf = X_pd_tfidf[0:df_train.shape[0],:]
    X_test_pd_tfidf = X_pd_tfidf[df_train.shape[0]:,:]
    # concatenate text, phone brand, device model matrix
    X_train_text_count = np.hstack([X_train_text_count, X_train_pd_count])
    X_test_text_count = np.hstack([X_test_text_count, X_test_pd_count])
    X_train_text_tfidf = np.hstack([X_train_text_tfidf, X_train_pd_tfidf])
    X_test_text_tfidf = np.hstack([X_test_text_tfidf, X_test_pd_tfidf])

    # numeric phone_brand
    df = pd.concat([df_train['phone_brand_en'], df_test['phone_brand_en']])
    a = pd.factorize(df)
    phone_brand_labels = a[1]
    df_train['phone_brand_en_num'] = a[0][0:df_train.shape[0]]
    df_test['phone_brand_en_num'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

    # numeric device_model
    df = pd.concat([df_train['device_model_en'], df_test['device_model_en']])
    a = pd.factorize(df)
    device_model_labels = a[1]
    df_train['device_model_en_num'] = a[0][0:df_train.shape[0]]
    df_test['device_model_en_num'] = a[0][df_train.shape[0]:df_train.shape[0]+df_test.shape[0]]

    # numeric group, create submission columns
    a = pd.factorize(df_train['group'], sort=True)
    df_train['group'] = a[0]
    group_labels = a[1]
    global group_list
    group_list = list(group_labels.values)
    for l in group_list:
        df_test[l] = 0

    a=df=None
    gc.collect()
    logger.info('finish numericing, vectorizing attributes')
    return df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, X_test_text_count

if __name__=='__main__':
    logging.info('logging_test')
    config_path = sys.argv[1]
    config_name = sys.argv[2]
    fill_na_opt = sys.argv[3]
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels = config.get(config_name, 'labels').split(',')
    clf_name = config.get(config_name, 'clf')

    logger.info('reading data')
    global group_list, df_test
    df_train = pd.read_csv('data/train_text_China.csv', dtype={'device_id':str})
    df_test = pd.read_csv('data/test_text_China.csv', dtype={'device_id':str})

    df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, \
            X_test_text_count = preprocess_data(df_train, df_test, fill_na_opt)

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
        clf = RandomForestClassifier(n_jobs=8, n_estimators=500)
    elif clf_name=='xgb':
        #subsample=0.5, colsample_bytree=0.5, colsample_bylevel=0.9
        clf = xgb.XGBClassifier(nthread=8, n_estimators=500, max_depth=10)

    train_model_with_feature(config_name, clf_name, clf, X, X_test, y)
