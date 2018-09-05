import xgboost as xgb
# Using https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
class XGBoostGbtreeAlgorithm:
    def __init__(self):
        self.model = None
        self.num_round = 250
        self.param = {'max_depth': 30, 'eta': 0.01, 'silent': 0, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}       

    def fit(self, X_train, y_train, X_valid, y_valid):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        evallist = [(dvalid, 'eval')]
        self.model = xgb.train(self.param, dtrain, self.num_round, evallist, early_stopping_rounds=3, verbose_eval=0)

        return self
    
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit).round()
    
    def save(self, drive_csv):
        self.model.save_model(drive_csv)