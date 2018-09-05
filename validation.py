from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from winner import Winner
import numpy as np
import pandas as pd
from algorithm_runner import AlgorithmRunner
from tqdm import tqdm
import os

class Validation:
    def run(drive_csv):
        scores = []
        grid_search_df = pd.read_csv('grid_search.csv')
        (X, y) = AlgorithmRunner.prepare_train(drive_csv)
        kf = KFold(n_splits=10, random_state=42)

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
            scores.append(AlgorithmRunner.metric(np.array(y_pred).round(), y_test))
        return np.average(scores)
    
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
                    results.append({'drive_csv': drive_csv, 'winner_f_beta': Validation.run(drive_csv)})
                    pbar.update(1)
            df = pd.DataFrame(results)
            df.to_csv(validation_file, index=False)
            return df