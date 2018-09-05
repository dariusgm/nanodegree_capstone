import lightgbm as lgb

# Using https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
# Use "same" values as xgboost for better comparing both
class LightGBMAlgorithm:
    def __init__(self):
        self.model = None
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 30,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

    def fit(self, X_train, y_train, X_valid, y_yalid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_yalid, reference=lgb_train)
       
        self.model = lgb.train(self.params,
                lgb_train,
                verbose_eval=False,
                num_boost_round=250,
                valid_sets=lgb_eval,
                early_stopping_rounds=3)

        return self
    
    def predict(self, X_test):
        # y_pred = self.model.predict(X_valid, num_iteration=self.model.best_iteration)
        return self.model.predict(X_test, num_iteration=self.model.best_iteration).round()
    
    def save(self, drive_csv):
        self.model.save_model(drive_csv)