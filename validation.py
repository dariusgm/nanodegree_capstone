from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from winner import Winner
import numpy as np
import pandas as pd
from algorithm_runner import AlgorithmRunner
from tqdm import tqdm
import os
from joblib import Parallel, delayed

class Validation:
    def merge_and_show(validation_results):
        merged_result = Winner.calculate()
        unneded_columnns = ['AdaBoostClassifier', 'GaussianNB', 'KerasAlgorithm', 'LightGBMAlgorithm', 'LinearSVC', 
                            'MLPClassifier', 'NearestCentroid', 'RandomForestClassifier' , 'SGDClassifier', 'SVC', 
                            'XGBoostGbtreeAlgorithm', 'winner', 'failure', 'file', 'lines', 'ok', 'size', 'test', 'train', 'validate']
        validation_results_mean = []
        for _i, row in validation_results.iterrows():
            values = [row['fbeta_run_0'], row['fbeta_run_1'], row['fbeta_run_2'], row['fbeta_run_3'], row['fbeta_run_4'], row['fbeta_run_5']]
            minimal_fbeta = min(values)
            maximal_fbeta =  max(values)
            mean_fbeta = np.mean(values)
            validation_results_mean.append({
                'drive_csv': row['drive_csv'],
                 'minimal_fbeta': minimal_fbeta,
                 'maximal_fbeta': maximal_fbeta,
                 'mean_fbeta': mean_fbeta
                }
            )

        kfold = merged_result.merge(pd.DataFrame(validation_results_mean), on='drive_csv')
        return kfold[kfold['winner'] == 'DecisionTreeClassifier'].drop(unneded_columnns,  axis='columns')

    def run(drive_csv):
        scores = {}
        grid_search_df = pd.read_csv('grid_search.csv')
        (X, y) = AlgorithmRunner.prepare_train(drive_csv)
        
        for iteration, random_state in enumerate([26,15,42,2,6,24]):
            kf = KFold(n_splits=10, random_state=random_state)
            mean_by_state = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                params = grid_search_df[grid_search_df['drive_csv'] == drive_csv].iloc[0].to_dict()
                del params['drive_csv']
                del params['fbetascore_grid']
                if params['min_samples_split'] > 1.:
                    params['min_samples_split'] = int(params['min_samples_split'])
                clf = DecisionTreeClassifier(**params)
                clf.fit(X_train, y_train) 
                y_pred = clf.predict(X_test)
                mean_by_state.append(AlgorithmRunner.metric(np.array(y_pred).round(), y_test))
            scores['fbeta_run_{}'.format(iteration)] =  np.mean(mean_by_state)
        return scores

    
    def run_all():
        validation_file = 'validation.csv'
        if os.path.exists(validation_file):
            return pd.read_csv(validation_file)
        else:
            merged_result = Winner.calculate()
            all_drives = list(merged_result[merged_result['winner'] == 'DecisionTreeClassifier']['drive_csv'])
            results = []
            with tqdm(total=len(all_drives)) as pbar:
                for drive_csv in all_drives:
                    pbar.set_description(drive_csv)
                    drive_information = {'drive_csv': drive_csv}
                    drive_information.update(Validation.run(drive_csv))
                    results.append(drive_information)
                    pbar.update(1)
            df = pd.DataFrame(results)
            df.to_csv(validation_file, index=False)
            return df