import unittest
import pandas as pd
from HP_OPT_class import HP_OPT_class_code, plot_tools

class TestHPOptClass(unittest.TestCase):
    def setUp(self):
        self.x_train = pd.read_csv('/home/shahid/ML/Machine_learning/Higgs_challenge/testing.csv').iloc[:,0:-1]
        self.y_train_new = pd.read_csv('/home/shahid/ML/Machine_learning/Higgs_challenge/testing.csv').iloc[:,-1]

    def tearDown(self):
        # Clean up resources if needed
        pass

    def test_mlp_objective(self):
        optimizer = HP_OPT_class_code.HPOpt(self.x_train, self.y_train_new)
        trial = optimizer.optimize("mlp").trials[0]  # Run only one trial for testing

        # You can add assertions to check the correctness of the optimization
        self.assertIn('n_layers', trial.params)
        self.assertIn('learning_rate', trial.params)

    def test_xgboost_objective(self):
        optimizer = HP_OPT_class_code.HPOpt(self.x_train, self.y_train_new)
        trial = optimizer.optimize("xgboost").trials[0]  # Run only one trial for testing

        # You can add assertions to check the correctness of the optimization
        self.assertIn('n_estimators', trial.params)
        self.assertIn('alpha', trial.params)

    def test_cnn_objective(self):
        optimizer = HP_OPT_class_code.HPOpt(self.x_train, self.y_train_new)
        trial = optimizer.optimize("cnn").trials[0]  # Run only one trial for testing

        # You can add assertions to check the correctness of the optimization
        self.assertIn('filters', trial.params)
        self.assertIn('kernel_size', trial.params)

    def test_transformer_objective(self):
        optimizer = HP_OPT_class_code.HPOpt(self.x_train, self.y_train_new)
        trial = optimizer.optimize("transformer").trials[0]  # Run only one trial for testing

        # You can add assertions to check the correctness of the optimization
        self.assertIn('d_model', trial.params)
        self.assertIn('num_heads', trial.params)

if __name__ == '__main__':
    unittest.main()
