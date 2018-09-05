from sklearn.tree import DecisionTreeClassifier
from algorithm_runner import AlgorithmRunner
import pandas as pd
import itertools
from joblib import Parallel, delayed
import numpy as np

# I had to implement a grid search by my own, as it was not running on my environment
class SearchWithGrid:
    def run_nested(drive_csv, X_train, y_train, X_test, y_test, params):
        clf = DecisionTreeClassifier(**params)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return {'params': params, 'fbetascore': AlgorithmRunner.metric(np.array(y_pred).round(), y_test)}
    
    def run(drive_csv):
        (X_train, y_train) =  AlgorithmRunner.prepare_train(drive_csv)
        (X_test, y_test) = AlgorithmRunner.prepare_test(drive_csv)

        params = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 1,2,3,4,5],
            'min_samples_split': [2, 3,4,5, 0.1,0.2,0.3],
            
        }

        # see: https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
        keys, values = zip(*params.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        nested_results = Parallel(n_jobs=-1, backend='threading')(delayed(SearchWithGrid.run_nested)(drive_csv, X_train, y_train, X_test, y_test, params) for params in experiments)
        filtered = []
        for result in nested_results:
            if result['fbetascore'] > 0:
                filtered.append(result)
        
        return {'drive_csv': drive_csv, 'nested_results': filtered}

    def calculate_best(drives):
        best_score = {}
        best_params = {}
        for result in drives:
            for nested_result in result['nested_results']:
                drive_csv = result['drive_csv']
                fbeta_score = nested_result['fbetascore']
                params = nested_result['params']
                if drive_csv in best_score:
                    if best_score[drive_csv] < fbeta_score:
                        best_score[drive_csv] = fbeta_score
                        best_params[drive_csv] = params
                else:
                    best_score[drive_csv] = fbeta_score
                    best_params[drive_csv] = params
        
        SearchWithGrid.save(best_params, best_score)       

    def save(best_params, best_score):
        tmp = []
        for drive_csv, params in best_params.items():
            criterion = params['criterion']
            splitter = params['splitter']
            min_sample_split = params['min_samples_split']
            fbeta_score = best_score[drive_csv]
            tmp.append({'drive_csv': drive_csv, 'fbetascore_grid': fbeta_score, 'criterion': criterion, 'splitter': splitter, 'min_samples_split' : min_sample_split})

        pd.DataFrame(tmp).to_csv('grid_search.csv', index=False)