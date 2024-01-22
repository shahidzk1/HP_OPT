import unittest
import pandas as pd
from HP_OPT_class import plot_tools
from HP_OPT_class_code import HPOpt

class TestHPOptClass(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('/home/shahid/ML/Machine_learning/Higgs_challenge/testing.csv')
        self.x_train = data.iloc[:, 0:-1]
        self.y_train_new = data.iloc[:, -1].replace({'s': 1, 'b': 0})

    def test_xgboost_objective(self):
        y_train_new = self.y_train_new.replace('s', 1, regex=True)
        y_train_new = self.y_train_new.replace('b', 0, regex=True)
        optimizer = HPOpt(self.x_train, y_train_new, tree_method_xgb='auto')
        trial = optimizer.optimize("xgboost").trials[0]
        self.assertIn('n_estimators', trial.params)
        self.assertIn('alpha', trial.params)

if __name__ == '__main__':
    unittest.main()
