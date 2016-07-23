from model_clf import *

# TODO
def xgb_parameter_search(X, X_test, y):
    # parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5], 'max_depth':range(3,15)}
    parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5],\
            'learning_rate':np.linespace(0.01,0.2,20)}
