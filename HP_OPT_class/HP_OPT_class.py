import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
import os
import gc
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

class HP_OPT:
    def __init__(self, x_train, y_train, n_classes, batch_size, n_trials,
                 MLP_hyp_par, xgb_hyp_para, cnn_hyp_par=None):
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.MLP_hyp_par = MLP_hyp_par
        self.xgb_hyp_para = xgb_hyp_para
        self.cnn_hyp_par = cnn_hyp_par
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=324, stratify=y_train
        )
        self.epochs = 10
        self.input_shape = (self.X_train.shape[1], 1)

    def keras_objective(self, trial):
        n_layers = trial.suggest_int("n_layers", self.MLP_hyp_par['n_layers'][0], self.MLP_hyp_par['n_layers'][1])
        model = Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.X_train.shape[1],)))
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), self.MLP_hyp_par['n_units'][0], self.MLP_hyp_par['n_units'][1], log=True)
            model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))
        model.add(tf.keras.layers.Dense(self.n_classes, activation=tf.nn.softmax))

        learning_rate = trial.suggest_float("learning_rate", self.MLP_hyp_par['learning_rate'][0], self.MLP_hyp_par['learning_rate'][1], log=True)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=tf.keras.metrics.AUC(),
        )

        tf.keras.backend.clear_session()

        model.fit(self.X_train, self.Y_train, batch_size=self.batch_size,
            callbacks=[optuna.integration.TFKerasPruningCallback(trial, "val_auc"), MyCustomCallback()],
            epochs=self.epochs,
            validation_split=0.1,
            verbose=1
        )

        score = model.evaluate(self.X_valid, self.Y_valid, verbose=0)
        return score[1]

    def xgboost_objective(self, trial):
        param = {
            'booster': 'gbtree',
            "n_estimators": trial.suggest_int("n_estimators", self.xgb_hyp_para['n_estimators'][0], self.xgb_hyp_para['n_estimators'][1], step=10),
            "alpha": trial.suggest_int("alpha", self.xgb_hyp_para['alpha'][0], self.xgb_hyp_para['alpha'][1]),
            "gamma": trial.suggest_float("gamma", self.xgb_hyp_para['gamma'][0], self.xgb_hyp_para['gamma'][1]),
            "learning_rate": trial.suggest_float("learning_rate", self.xgb_hyp_para['learning_rate'][0], self.xgb_hyp_para['learning_rate'][1], log=True),
            "max_depth": trial.suggest_int("max_depth", self.xgb_hyp_para['max_depth'][0], self.xgb_hyp_para['max_depth'][1]),
            'objective': 'multi:softmax',
            'num_class': len(np.unique(self.y_train)),
        }

        clf = XGBClassifier(**param, tree_method='gpu_hist')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = cross_val_score(clf, self.x_train, self.y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        auc_mean = auc_scores.mean()

        trial.set_user_attr("n_estimators", clf.n_estimators)

        return auc_mean

    def cnn_objective(self, trial):
      scaler = StandardScaler()
      X_train = scaler.fit_transform(self.X_train)
      X_valid = scaler.transform(self.X_valid)
      X_train = self.X_train.to_numpy().reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
      X_valid = self.X_valid.to_numpy().reshape(self.X_valid.shape[0], self.X_valid.shape[1], 1)

      clear_session()

      model = Sequential()
      model.add(
          Conv1D(
              filters=trial.suggest_categorical("filters", self.cnn_hyp_par['filters']),
              kernel_size=trial.suggest_categorical("kernel_size", self.cnn_hyp_par['kernel_size']),
              strides=trial.suggest_categorical("strides", self.cnn_hyp_par['strides']),
              activation="linear", input_shape=self.input_shape,
          )
      )

      n_conv_layers = trial.suggest_int("n_conv_layers", self.cnn_hyp_par['n_conv_layers_min'], self.cnn_hyp_par['n_conv_layers_max'])

      for i in range(n_conv_layers - 1):
          model.add(
              Conv1D(
                  filters=trial.suggest_categorical("filters", self.cnn_hyp_par['filters']),
                  kernel_size=trial.suggest_categorical("kernel_size", self.cnn_hyp_par['kernel_size']),
                  strides=trial.suggest_categorical("strides", self.cnn_hyp_par['strides']),
                  activation="linear",
              )
          )
          dropout_rate = trial.suggest_float("conv_dropout_rate_l{}".format(i), self.cnn_hyp_par['dropout_min'], self.cnn_hyp_par['dropout_max'])
          model.add(Dropout(dropout_rate))

      model.add(Flatten())

      n_layers = trial.suggest_int("n_layers", self.cnn_hyp_par['n_layers_min'], self.cnn_hyp_par['n_layers_max'])
      for i in range(n_layers):
          num_hidden = trial.suggest_int("n_units_l{}".format(i), self.cnn_hyp_par['n_units_min'], self.cnn_hyp_par['n_units_max'], log=True)
          model.add(Dense(num_hidden, activation="relu", activity_regularizer=l1(0.001)))
          dropout_rate = trial.suggest_float("dropout_rate_l{}".format(i), self.cnn_hyp_par['dropout_min'], self.cnn_hyp_par['dropout_max'])
          model.add(Dropout(dropout_rate))
          model.add(BatchNormalization())

      model.add(Dense(self.n_classes, activation="softmax"))

      learning_rate = trial.suggest_float("learning_rate", self.cnn_hyp_par['learning_rate_min'], self.cnn_hyp_par['learning_rate_max'], log=True)
      model.compile(
          loss="sparse_categorical_crossentropy",
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Use tf.keras.optimizers
          metrics=["accuracy"],
      )

      early_stop = EarlyStopping(monitor='val_loss', patience=3)
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

      model.fit(
          X_train, self.Y_train, validation_data=(X_valid, self.Y_valid),
          shuffle=True, batch_size=self.batch_size,
          epochs=self.epochs, verbose=False, callbacks=[early_stop, reduce_lr]
      )

      score = model.evaluate(X_valid, self.Y_valid, verbose=0)
      print(f"Trial {trial.number}, Score: {score[1]}")
      print(f"Parameters: {trial.params}")

      return score[1]

    def optimize(self, model_type):
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(), sampler=optuna.samplers.TPESampler())

        if model_type == "keras":
            study.optimize(self.keras_objective, n_trials=self.n_trials, gc_after_trial=True)
            print("Keras Best Trial:")
        elif model_type == "xgboost":
            study.optimize(self.xgboost_objective, n_trials=self.n_trials, gc_after_trial=True)
            print("XGBoost Best Trial:")
        elif model_type == "cnn":
            study.optimize(self.cnn_objective, n_trials=self.n_trials, gc_after_trial=True)
            print("CNN Best Trial:")
        else:
            raise ValueError("Invalid model_type. Choose 'keras', 'xgboost', or 'cnn'.")

        return study
