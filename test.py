MSE = []
RMSE = []
R_squared_validation = []
R_squared_test = []
feature_importance = []

for i in range(12):
    y = Y[Y.columns[i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
    
    params = {
        "max_depth":[2,3,4,5],
        'splitter':["random", "best"],
        'min_samples_leaf':[1,2,3,4],
        'min_samples_split':[2,3,4,5]
    }
    
    DT_regression = DecisionTreeRegressor()
    model = GridSearchCV(DT_regression, param_grid=params, cv=5)
    model.fit(X_train, y_train)
    max_depth= model.best_params_["max_depth"]
    splitter = model.best_params_["splitter"]
    min_samples_leaf= model.best_params_["min_samples_leaf"]
    min_samples_split = model.best_params_["min_samples_split"]
    

    DT_regression = DecisionTreeRegressor(random_state=0, max_depth=max_depth, splitter=splitter, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    cv_score = cross_validate(DT_regression       #实例化的模型
				, X   #完整的特征值
				, y #完整的目标值
				, cv=10         #几折交叉验证
				,scoring = ["neg_mean_squared_error","neg_mean_squared_log_error","r2"]   
				)

    
    
    MSE.append(cv_score["test_neg_mean_squared_error"].mean())
    RMSE.append(cv_score["test_neg_mean_squared_log_error"].mean())
    R_squared_validation.append(cv_score["test_r2"].mean())

    DT_regression = DecisionTreeRegressor(random_state=0, max_depth=max_depth, splitter=splitter, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    DT_regression.fit(X_train,y_train)
    R_squared_test.append(DT_regression.score(X_test, y_test))
    joblib.dump(DT_regression, "model/DT_optimized_regression/model{}.pkl".format(i+1))

    feature_importance.append(list(DT_regression.feature_importances_))

