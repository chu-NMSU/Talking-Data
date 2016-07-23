from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

def xgb_parameter_search(X, X_test, y):
    # parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5], 'max_depth':range(3,15)}
    parameters = {'objective':['multi:softprob'], 'nthread':[1], 'n_estimators':[1000], 'subsample':[0.5],\
        'learning_rate':np.linespace(0.01,0.2,20)}
    # TODO

def rf_parameter_search(X, X_test, y):
    rf = RandomForestClassifier()
    parameters={'max_depth':range(3,15), 'min_samples_leaf':range(5,20)}
    clf = GridSearchCV(rf, parameters, cv=5, n_jobs=5)
    print 'grid scores=', clf.grid_scores_
    print 'best estimator=', clf.best_estimator_
    print 'best score=', clf.best_score_
    print 'best parameters=', clf.best_params_

