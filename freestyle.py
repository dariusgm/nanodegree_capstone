import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from winner import Winner
class Freestyle:
    def __init__(self):
        self.merged_result = Winner.calculate()
        self.winners = []
        self.algorithms = ['AdaBoostClassifier', 'DecisionTreeClassifier', 'FakeAlgorithm',
               'GaussianNB', 'KerasAlgorithm', 'LightGBMAlgorithm', 'LinearSVC',
               'MLPClassifier', 'NearestCentroid', 'RandomForestClassifier',
               'SGDClassifier', 'SVC', 'XGBoostGbtreeAlgorithm']
        for algo in self.algorithms:
            total = 0
            for _i, row in self.merged_result.iterrows():
                if row['winner'] == algo:
                    total += 1
            self.winners.append(total)

        self.algorithm_short = [x.replace('Classifier', '').replace('Algorithm', '') for x in self.algorithms]

    def plot(self):
        # as suggest by review, changed to bar charts
        # the nice colors: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        trace = go.Bar(
                    marker=dict(
                        color=['#e6194b',
                               '#3cb44b',
                               '#ffe119',
                               '#4363d8',
                               '#f58231',
                               '#911eb4',
                               '#46f0f0',
                               '#f032e6',
                               '#bcf60c',
                               '#fabebe',
                               '#008080',
                               '#e6beff',
                               '#9a6324'], line=dict(color='#000000', width=2)),
                    textfont=dict(size=10),
                    x=self.algorithm_short,
                    y=self.winners
            )

        plotly.offline.iplot({
            'data': [trace],
            'layout': go.Layout(title="Total Algorithm winners", xaxis=dict(tickangle=-45), autosize=False, width=600, height=400)
        })