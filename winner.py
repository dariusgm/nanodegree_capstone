import glob
import pandas as pd
import os

class Winner():
    def calculate():
        if os.path.exists('merged_result.csv'):
            return pd.read_csv('merged_result.csv')
        else:
            all_results = {}
            for d in glob.glob('*_models'):
                for file in glob.glob('{}/*.csv.csv'.format(d)):
                    r = pd.read_csv(file)
                    data = r.iloc[0]
                    drive = data['drive']
                    algorithm = data['clf_name']
                    f_beta_score = data['f_beta_score']
                    runtime = data['time']
                    if not drive in all_results:
                        all_results[drive] = {}
                        all_results[drive][algorithm] = f_beta_score
                    else:
                        all_results[drive][algorithm] = f_beta_score 

            final_results = []
            # Move every result to own Column
            for drive_csv, h in all_results.items():
                nested_result = {'drive_csv': drive_csv}
                for algorithm, f_beta_score in h.items():
                    nested_result[algorithm] = f_beta_score
                final_results.append(nested_result)

            results_df = pd.DataFrame(final_results)
            # Fill NaN with 0 for algorithms that where not run for this dataset
            results_df.fillna(0.0, inplace=True)
            qualified_df = pd.read_csv('qualified.csv')

            # Combine with qualified drives
            merged_result = results_df.merge(qualified_df, on='drive_csv')

            # Calculate the winner per drive and append it to the present data
            tmp = []
            for _i, row in merged_result.iterrows():
                local_winner = ''
                local_winner_f_beta_score = 0
                for row_name in ['AdaBoostClassifier', 'DecisionTreeClassifier', 'FakeAlgorithm','GaussianNB', 'KerasAlgorithm', 'LightGBMAlgorithm', 'LinearSVC','MLPClassifier', 'NearestCentroid', 'RandomForestClassifier','SGDClassifier', 'SVC', 'XGBoostGbtreeAlgorithm']:
                    if row[row_name] > local_winner_f_beta_score:
                        local_winner_f_beta_score = row[row_name]
                        local_winner = row_name
                tmp.append({'drive_csv': row['drive_csv'], 'winner': local_winner, 'winner_f_beta': local_winner_f_beta_score})

            merged_result = merged_result.merge(pd.DataFrame(tmp), on='drive_csv')
            merged_result.sort_values('winner_f_beta', inplace=True)
            merged_result.to_csv('merged_result.csv', index=False)
            return merged_result