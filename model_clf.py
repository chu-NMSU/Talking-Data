from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import sys
import os
import json
import time
import logging
import datetime
import ConfigParser
from parameter_tune import *

PATH = 'data' # China data is under 'data/China'
# create logger
logger = logging.getLogger('Talking-Data-model')
logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

def train_model_with_feature(config_name, clf_name, fill_na_opt, PCA_n_comp, clf, X, X_test, y):
    if PCA_n_comp!=-1:
        pca = PCA(PCA_n_comp) #PCA dimension reduction
        logger.info('PCA fit on count matrix')
        # rescale num to (0,1)
        X_all = pca.fit_transform( minmax_scale(np.vstack([X, X_test])) )
        X, X_test = X_all[:X.shape[0], :], X_all[X.shape[0]:, :]
        logger.info('PCA fit done')

    logger.info('start training')
    print 'training size', X.shape, 'test size', X_test.shape
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9)
    if clf_name=='xgb':
        clf.fit(X_train,y_train,eval_metric='mlogloss')
    else:
        clf.fit(X_train,y_train)
    logger.info(clf_name+'-'+fill_na_opt+'-pca('+str(PCA_n_comp)+') train log-loss='\
            +str(log_loss(y_train, clf.predict_proba(X_train))))
    logger.info(clf_name+'-'+fill_na_opt+'-pca('+str(PCA_n_comp)+') validate log-loss='\
            +str(log_loss(y_val, clf.predict_proba(X_val))))

    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    df_test[group_list] = y_pred
    logger.info('finish training')
    # , 'phone_brand_en', 'device_model_en'
    df_test.to_csv('output/'+config_name+'-'+clf_name+'-'+fill_na_opt+'-pca'+\
            str(PCA_n_comp)+'-'+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))\
            +'.csv', columns=['device_id']+group_list, index=False)
    logger.info('finish outputing result')

def fill_na_test(df_train, df_test, X_train_text_tfidf, X_test_text_tfidf, \
        X_train_text_count, X_test_text_count, option):
    logger.info('start to fill na data')
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
                    X_test_text_count[i,:]=X_train_text_count[group.index,:].mean(axis=0)

    ## save filled matrix
    np.savetxt(PATH+'/X_test_text_count-'+option+'.csv', X_test_text_count, delimiter=',')
    np.savetxt(PATH+'/X_test_text_tfidf-'+option+'.csv', X_test_text_tfidf, delimiter=',')
    np.savetxt(PATH+'/X_train_text_count-'+option+'.csv', X_train_text_count, delimiter=',')
    np.savetxt(PATH+'/X_train_text_tfidf-'+option+'.csv', X_train_text_tfidf, delimiter=',')
    logger.info('finish filling na data')

    return X_test_text_tfidf, X_test_text_count

def preprocess_data(df_train, df_test, fill_na_opt):
    logger.info('start preprocessing data')

    ## text feature engineering
    df_train['text'] = df_train['text'].str.lower()
    df_test['text'] = df_test['text'].str.lower()
    df_test['text'] = df_test['text'].fillna('')

    X_test_text_count=X_test_text_tfidf=X_train_text_count=X_train_text_tfidf=None

    logger.info('start vectorizing data')
    if os.path.exists(PATH+'/X_test_text_count-'+fill_na_opt+'.csv') and \
            os.path.exists(PATH+'/X_test_text_tfidf-'+fill_na_opt+'.csv') and \
            os.path.exists(PATH+'/X_train_text_count-'+fill_na_opt+'.csv') and \
            os.path.exists(PATH+'/X_train_text_tfidf-'+fill_na_opt+'.csv'):
        X_test_text_count = np.loadtxt(PATH+'/X_test_text_count-'+fill_na_opt+'.csv', \
                delimiter=',')
        X_test_text_tfidf = np.loadtxt(PATH+'/X_test_text_tfidf-'+fill_na_opt+'.csv', \
                delimiter=',')
        X_train_text_count = np.loadtxt(PATH+'/X_train_text_count-'+fill_na_opt+'.csv', \
                delimiter=',')
        X_train_text_tfidf = np.loadtxt(PATH+'/X_train_text_tfidf-'+fill_na_opt+'.csv', \
                delimiter=',')
    else:
        df = pd.concat([df_train['text'].str.replace('[^a-zA-Z ]',' ')+\
            ' '+df_train['phone_brand_en'].str.replace('[^a-zA-Z ]',' ')+\
            ' '+df_train['device_model_en'].str.replace('[^a-zA-Z ]',' '), \
                df_test['text'].str.replace('[^a-zA-Z ]',' ')+\
            ' '+df_test['phone_brand_en'].str.replace('[^a-zA-Z ]',' ')+\
            ' '+df_test['device_model_en'].str.replace('[^a-zA-Z ]',' ')])
        count_vect = CountVectorizer() #word count vectorization
        X_text_counts = count_vect.fit_transform(df).toarray()
        X_train_text_count = X_text_counts[0:df_train.shape[0],:]
        X_test_text_count = X_text_counts[df_train.shape[0]:,:]
        with open(PATH+'/count_vocab.json', 'w') as outfile:
            json.dump(count_vect.vocabulary_, outfile, indent=1, sort_keys=True)

        tfidf_vect = TfidfVectorizer() #tfidf count vectorization
        X_text_tfidf = tfidf_vect.fit_transform(df).toarray()
        X_train_text_tfidf = X_text_tfidf[0:df_train.shape[0],:]
        X_test_text_tfidf = X_text_tfidf[df_train.shape[0]:,:]
        with open(PATH+'/tfidf_vocab.json', 'w') as outfile:
            json.dump(tfidf_vect.vocabulary_, outfile, indent=1, sort_keys=True)

        X_test_text_tfidf, X_test_text_count = fill_na_test(df_train, df_test, \
                X_train_text_tfidf, X_test_text_tfidf, X_train_text_count, \
                X_test_text_count, fill_na_opt)

    logger.info('finish vectorizing data')

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
    if len(sys.argv)<5:
        print 'python model_clf.py config_path comfig_name fill_na_opt PCA_n_comp <parameter_tune_opt>'
    config_path = sys.argv[1]
    config_name = sys.argv[2]
    fill_na_opt = sys.argv[3]
    PCA_n_comp = int(sys.argv[4])
    parameter_tune = ''
    if len(sys.argv)>=6:
        parameter_tune = sys.argv[5]

    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels = config.get(config_name, 'labels').split(',')
    clf_name = config.get(config_name, 'clf')

    logger.info('reading data')
    global group_list, df_test
    df_train = pd.read_csv(PATH+'/train_text.csv', dtype={'device_id':str})
    df_test = pd.read_csv(PATH+'/test_text.csv', dtype={'device_id':str})

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
        clf = MultinomialNB(alpha=0.001, fit_prior=True)
    elif clf_name=='lr':
        clf = LogisticRegression()
    elif clf_name=='rf':
        clf = RandomForestClassifier(n_jobs=4, n_estimators=1000, max_depth=12, \
                min_samples_leaf=17)
    elif clf_name=='xgb':
        clf = xgb.XGBClassifier(objective='multi:softprob', nthread=8, n_estimators=1000,\
            max_depth=10, silent=False, subsample=0.8, colsample_bytree=0.5)

    if parameter_tune=='rf':
        logger.info('start randome forest parameter grid search')
        rf_parameter_search(X, X_test, y)
    else:
        train_model_with_feature(config_name, clf_name, fill_na_opt, PCA_n_comp, clf, X, X_test, y)

