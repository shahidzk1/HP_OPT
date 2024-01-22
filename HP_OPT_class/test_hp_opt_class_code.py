import unittest
import pandas as pd
import numpy as np
from HP_OPT_class import plot_tools
from HP_OPT_class_code import HPOpt

class TestHPOptClass(unittest.TestCase):
    def setUp(self):
        #change the path to your local directory
        path = '/home/shahid/ML/Machine_learning/Higgs_challenge/testing.csv'
        data = pd.read_csv(path)
        self.x_train = data.iloc[:, 0:-1]
        self.y_train_new = data.iloc[:, -1].replace({'s': 1, 'b': 0})
        self.y_train_new = self.y_train_new.replace('s', 1, regex=True)
        self.y_train_new = self.y_train_new.replace('b', 0, regex=True)

    def test_xgboost_objective(self):
        optimizer = HPOpt(self.x_train, self.y_train_new, tree_method_xgb='auto', n_trials=1)
        trial = optimizer.optimize("xgboost").trials[0]
        self.assertIn('n_estimators', trial.params)
        self.assertIn('alpha', trial.params)
        print("The XGBoost Optuna testing unit works well")

    def test_mlp_objective(self):
        keras_hyperparameters = {
            'n_layers': (2, 3),
            'n_units': (1, 3),
            'learning_rate': (1e-5, 1e-1)
        }
        optimizer = HPOpt(self.x_train, self.y_train_new, batch_size=1024, n_trials=1, n_epochs = 1,
                          mlp_hyp_par=keras_hyperparameters, num_classes=len(np.unique(self.y_train_new)),)
        trial = optimizer.optimize("mlp").trials[0]
        self.assertIn('n_layers', trial.params)
        self.assertIn('learning_rate', trial.params)
        print("The MLP Optuna testing unit works well")
    
    def test_cnn_objective(self):
        cnn_hyperparameters = {
        'filters': [32, 64],
        'kernel_size': [3, 5],
        'strides': [1, 2],
        'n_conv_layers_min': 0,
        'n_conv_layers_max': 3,
        'dropout_min': 0.0,
        'dropout_max': 0.5,
        'n_layers_min': 2,
        'n_layers_max': 22,
        'n_units_min': 1000,
        'n_units_max': 2500,
        'learning_rate_min': 1e-5,
        'learning_rate_max': 1e-1
    }
        optimizer = HPOpt(self.x_train, self.y_train_new, batch_size=1024, n_trials=1, n_epochs = 1,
                          cnn_hyp_par=cnn_hyperparameters,
                                        num_classes=len(np.unique(self.y_train_new)))
        trial = optimizer.optimize("cnn").trials[0]
        self.assertIn('filters', trial.params)
        self.assertIn('kernel_size', trial.params)
        print("The CNN Optuna testing unit works well")

    def test_transformer_objective(self):
        transformer_hyp_par = {'d_model': (64, 256),
                               'num_heads': (2, 8),
                               'num_layers': (2, 15),
                               'dropout': (0.1, 0.5),
                               'weight_decay': (1e-6, 1e-3),
                               'l1_regularization': (1e-6, 1e-3),
                               'learning_rate': (1e-5, 1e-2)}
        optimizer = HPOpt(self.x_train,
                          self.y_train_new,
                          batch_size=1024,
                          n_trials=1,
                          n_epochs = 1,
                          transformer_hyp_par=transformer_hyp_par,
                          num_classes=len(np.unique(self.y_train_new)))
        trial = optimizer.optimize("transformer").trials[0]
        self.assertIn('d_model', trial.params)
        self.assertIn('num_heads', trial.params)
        print("The transformer Optuna testing unit works well")

if __name__ == '__main__':
    unittest.main()
    print("All tests passed successfully")
