from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def xgb_parameter_search(X, X_test, y):
    # parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5], 'max_depth':range(3,15)}
    parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5],\
        'learning_rate':np.linespace(0.01,0.2,20)}
    # TODO

def rf_parameter_search(X, X_test, y):
    rf = RandomForestClassifier()
    parameters={'max_depth':range(3,15), 'min_samples_leaf':range(5,20)}
    clf = GridSearchCV(rf, parameters, cv=5, n_jobs=5)
    clf.fit(X,y)
    print 'grid scores=', clf.grid_scores_
    print 'best estimator=', clf.best_estimator_
    print 'best score=', clf.best_score_
    print 'best parameters=', clf.best_params_

    # best estimator= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #     max_depth=12, max_features='auto', max_leaf_nodes=None,
    #     min_samples_leaf=17, min_samples_split=2,
    #     min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
    #     oob_score=False, random_state=None, verbose=0,
    #     warm_start=False)
    # best score= 0.243199188612
    # best parameters= {'max_depth': 12, 'min_samples_leaf': 17}
