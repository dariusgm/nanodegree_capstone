from helper import Helper


import numpy as np
import pandas as pd
import os
from time import time
# see: https://docs.python.org/3/library/concurrent.futures.html
from concurrent.futures import Executor, ThreadPoolExecutor
from sklearn.metrics import fbeta_score
from sklearn.externals import joblib

class AlgorithmRunner():
    def metric(y_pred, y_true):
        """
         Keyword arguments:
         y_pred - a list or np.array object containing the predicted values from a machine learning algorithm.
         y_true - a list or np.array object containing the correct labels that should match `y_pred`.

         As y_pred can have no predictions at all, we handle this special case and return a score of `0.0`.
        """
        # Preventing calculation warning from fbeta_score
        if np.array(y_pred).sum() == 0.0 or np.array(y_true).sum() == 0.0:
            return 0.0
        else:
            return fbeta_score(y_true, y_pred, average='macro', beta=1)

    def build_dtype(model_csv):
        """
        create a dict where each column is mapped to the np.float64 datatype, to make sure that the data is correcly read
        """
        with open(model_csv) as f:
            first_line = f.readline()
            parts = first_line.split(',')
            result = {}
            for p in parts:
                result[p] = np.float64

            return result       

    def prepare_data(X):
        """Cleans up the readed data in a general way. Called by prepare_* methods"""
        helper = Helper()
        X.dropna(axis='columns', inplace=True, how='all')
        # this is needed as some values could not be extracted
        X.dropna(axis='index', inplace=True, how='any')
        X.rename(helper.smart_to_name(), inplace=True)
        X.reset_index() # required?, see:https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
        y = X['failure']
        X.drop(labels=['failure'], axis='columns', inplace=True)
        return (X, y)

    def prepare_train(model_csv):
        train_file = os.path.join('train', model_csv)
        X_train = pd.read_csv(train_file, float_precision='high', dtype=AlgorithmRunner.build_dtype(train_file))
        return AlgorithmRunner.prepare_data(X_train)

    def prepare_test(model_csv):
        test_file = os.path.join('test', model_csv) 
        X_test = pd.read_csv(test_file, float_precision='high', dtype=AlgorithmRunner.build_dtype(test_file))
        return AlgorithmRunner.prepare_data(X_test)

    def prepare_validate(model_csv):
        validate_file = os.path.join('validate', model_csv)
        X_validate = pd.read_csv(validate_file, float_precision='high', dtype=AlgorithmRunner.build_dtype(validate_file))
        return AlgorithmRunner.prepare_data(X_validate)

    def prepare_parallel(model_csv):
        """Reads train, test and validation set in parallel to reduce the runtime of the training and better use IO"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_train = executor.submit(AlgorithmRunner.prepare_train, model_csv)
            future_test = executor.submit(AlgorithmRunner.prepare_test, model_csv)
            future_validate = executor.submit(AlgorithmRunner.prepare_validate, model_csv)

            # Now join everything so make sure we can process in sync
            X_train, y_train = future_train.result()
            X_test, y_test = future_test.result()
            X_valid, y_valid = future_validate.result()

            assert (set(X_train.columns) == set(X_test.columns) and set(X_train.columns) == set(X_valid.columns))

            return (X_train, y_train, X_test, y_test, X_valid, y_valid)

    def run_algorithm(clf, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None):
        """Runs a given algorithm clf on the data. Calculates the FBeta Score of the result."""
        if X_valid is None:
            clf = clf.fit(X_train, y_train)
        else:
            clf = clf.fit(X_train, y_train, X_valid, y_valid) # use validation set for algorithm improvement if possible
        y_pred = clf.predict(X_test)


        return AlgorithmRunner.metric(np.array(y_pred).round(), y_test)
    
    def build_model_path(clf, drive):
        """Build the appropriate paths for model and calculated csv output according to the class that is running"""
        clf_name = clf.__class__.__name__
        dict_key = '{}-{}'.format(clf_name, drive)

        if 'Keras' in clf_name:
            model_path = os.path.join('keras_models', '{}.h5'.format(dict_key))
            result_path = os.path.join('keras_models', '{}.csv'.format(dict_key))
            return (model_path, result_path)
        elif 'XGBoost' in clf_name:
            model_path = os.path.join('xgboost_models', '{}.bin'.format(dict_key))
            result_path = os.path.join('xgboost_models', '{}.csv'.format(dict_key))
            return (model_path, result_path)
        elif 'LightGBM' in clf_name:
            model_path = os.path.join('lightgbm_models', '{}.bin'.format(dict_key))
            result_path = os.path.join('lightgbm_models', '{}.csv'.format(dict_key))
            return (model_path, result_path)
        else: 
            model_path = os.path.join('sklearn_models', '{}.pkl'.format(dict_key))
            result_path =  os.path.join('sklearn_models', '{}.csv'.format(dict_key))
            return (model_path, result_path)  
    
    # to better use cpu power, lets to it in parallel
    # see: https://stackoverflow.com/questions/29589327/train-multiple-models-in-parallel-with-sklearn/29596675
    # and see: https://pythonhosted.org/joblib/parallel.html
    def run_parallel(clf, drive, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None):
        """
        Assuming that the algorithm runs against other algorithms. Save generated model and metric data to a file.
        Restore the data when rerun
        """
        clf_name = clf.__class__.__name__
        dict_key = '{}-{}'.format(clf_name, drive)

        model_path, result_path = AlgorithmRunner.build_model_path(clf, drive)
        # Restore data to prevent a recalculation of the same data.
        if os.path.exists(model_path):
            data = pd.read_csv(result_path).to_dict()
            clf_name = data['clf_name'][0]
            drive = data['drive'][0]
            f_beta_score =  data['f_beta_score'][0]
            t = data['time'][0]
            return {'clf_name':clf_name, 'drive': drive, 'f_beta_score': f_beta_score, 'time': t}
        else:
            start = time()
            f_beta_score = AlgorithmRunner.run_algorithm(clf, X_train, y_train, X_test, y_test, X_valid, y_valid)
            finish = time() - start
            # https://stackoverflow.com/questions/5268404/what-is-the-fastest-way-to-check-if-a-class-has-a-function-defined
            # if present use custom save logic, otherwise use joblib
            if hasattr(clf, "save"):
                clf.save(model_path)
            else:
                # Dump clf for sklearn
                joblib.dump(clf, model_path)

            # dump results
            data = {'clf_name': clf_name, 'time': finish, 'f_beta_score': f_beta_score, 'drive': drive}
            pd.DataFrame([data]).to_csv(result_path, index=False)

        return data