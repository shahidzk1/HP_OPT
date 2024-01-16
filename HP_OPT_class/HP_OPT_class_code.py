import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical
import os
import gc
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda.amp as amp


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()

class HP_OPT:
    def __init__(self, x_train, y_train, batch_size=32, n_trials=10, n_epochs =10, mlp_hyp_par=None, xgb_hyp_para=None, cnn_hyp_par=None, transformer_hyp_par=None, num_classes=None):
        """"
        This class can be used to optimize the hyperparameters of different machine learning algorithms such as multi-layer perception, XGboost, Convolutional neural network (1D),
        and transformers for the classification of tabular data.
        
        Use case: 
        hp_optimizer = HP_OPT(x_train, y_train, batch_size=32, n_trials=10, mlp_hyp_par=None, xgb_hyp_para=None, cnn_hyp_par=None, transformer_hyp_par=None, num_classes=None)
        study_keras = hp_optimizer.optimize("keras")
        
        Args:
            x_train                (pandas.core.frame.DataFrame)    : Should contain variables for training
            y_train                (pandas.core.series.Series)      : The target array
            batch_size             (int)                            : The size of the batch for the neural networks
            n_trials               (int)                            : The number of trials for the Optuna HP optimization
            n_epochs               (int)                            : The number of epochs per trial of Optuna
            mlp_hyp_par            (dict)                           : A dictionary of the hyper-parameters of the MLP
            xgb_hyp_para           (dict)                           : A dictionary of the hyper-parameters of the XGBoost Classifier
            cnn_hyp_par            (dict)                           : A dictionary of the hyper-parameters of the CNN
            transformer_hyp_par    (dict)                           : A dictionary of the hyper-parameters of the transformer            
        
        """
        self.x_train = x_train
        self.y_train = y_train
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.n_trials = n_trials
        #The default hyperparameters are given here
        self.mlp_hyp_par = mlp_hyp_par or { 'n_layers': (2, 22),'n_units': (2500, 4500), 'learning_rate': (1e-5, 1e-1)}
        
        self.xgb_hyp_para = xgb_hyp_para or { 'n_estimators': (100, 1000), 'alpha': (2, 30),'gamma': (0, 1), 'learning_rate': (0.01, 1), 'max_depth': (0, 10) }

        self.cnn_hyp_par = cnn_hyp_par or {'filters': [32, 64], 'kernel_size': [3, 5],'strides': [1, 2],'n_conv_layers_min': 0,'n_conv_layers_max': 3,'dropout_min': 0.0,'dropout_max': 0.5,
        'n_layers_min': 2, 'n_layers_max': 22,'n_units_min': 1000,'n_units_max': 1500, 'learning_rate_min': 1e-5,'learning_rate_max': 1e-1}
                     
        self.transformer_hyp_par = transformer_hyp_par or {'d_model': (64, 256),'num_heads': (2, 8),'num_layers': (2, 15),'dropout': (0.1, 0.5),
        'weight_decay': (1e-6, 1e-3),'l1_regularization': (1e-6, 1e-3), 'learning_rate': (1e-5, 1e-2) }
                     
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split( x_train, y_train, test_size=0.2, random_state=324, stratify=y_train )
        self.epochs = n_epochs
        self.input_shape = (self.X_train.shape[1],)
        
    def keras_objective(self, trial):
        """"
        This method takes the hyperparameters of the MLP, creates a model and then trains and tests the model 
        """
      le = LabelEncoder()
      Y_train_encoded = le.fit_transform(self.Y_train)
      Y_valid_encoded = le.transform(self.Y_valid)
      Y_train_new = to_categorical(Y_train_encoded)
      Y_valid_new = to_categorical(Y_valid_encoded)
      n_layers = trial.suggest_int("n_layers", self.mlp_hyp_par['n_layers'][0], self.mlp_hyp_par['n_layers'][1])
      model = Sequential()
      model.add(tf.keras.layers.Flatten(input_shape=(self.X_train.shape[1],)))
      for i in range(n_layers):
          num_hidden = trial.suggest_int("n_units_l{}".format(i), self.mlp_hyp_par['n_units'][0], self.mlp_hyp_par['n_units'][1], log=True)
          model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))
      model.add(tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax))
      learning_rate = trial.suggest_float("learning_rate", self.mlp_hyp_par['learning_rate'][0], self.mlp_hyp_par['learning_rate'][1], log=True)
      model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
          metrics=tf.keras.metrics.AUC(),
      )
      tf.keras.backend.clear_session()
      model.fit(self.X_train, Y_train_new, batch_size=self.batch_size,
          callbacks=[optuna.integration.TFKerasPruningCallback(trial, "val_auc"), MyCustomCallback()],
          epochs=self.epochs,
          validation_split=0.1,
          verbose=1
      )
      score = model.evaluate(self.X_valid, Y_valid_new, verbose=0)
      return score[1]

    def xgboost_objective(self, trial):
        """"
        This method takes the hyperparameters of the XGBoost, creates a model and then trains and tests the model 
        """
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
        """"
        This method takes the hyperparameters of the CNN, creates a model and then trains and tests the model 
        """
      input_shape = (self.X_train.shape[1], 1)
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
              activation="linear", input_shape=input_shape, name=f'conv1d_{trial.number}' ,
          )
      )
      n_conv_layers = trial.suggest_int("n_conv_layers", self.cnn_hyp_par['n_conv_layers_min'], self.cnn_hyp_par['n_conv_layers_max'])
      for i in range(n_conv_layers - 1):
          model.add(
              Conv1D(
                  filters=trial.suggest_categorical("filters", self.cnn_hyp_par['filters']),
                  kernel_size=trial.suggest_categorical("kernel_size", self.cnn_hyp_par['kernel_size']),
                  strides=trial.suggest_categorical("strides", self.cnn_hyp_par['strides']),
                  activation="linear",  name=f'conv1d_{trial.number}_layer_{i}',
              )
          )
          dropout_rate = trial.suggest_float("conv_dropout_rate_l{}".format(i), self.cnn_hyp_par['dropout_min'], self.cnn_hyp_par['dropout_max'])
          model.add(Dropout(dropout_rate))
      model.add(Flatten())
      n_layers = trial.suggest_int("n_layers", self.cnn_hyp_par['n_layers_min'], self.cnn_hyp_par['n_layers_max'])
      for i in range(n_layers):
          num_hidden = trial.suggest_int("n_units_l{}".format(i), self.cnn_hyp_par['n_units_min'], self.cnn_hyp_par['n_units_max'], log=True)
          model.add(Dense(num_hidden, activation="relu", activity_regularizer=l1(0.001),name=f'dense_{trial.number}_layer_{i}'))
          dropout_rate = trial.suggest_float("dropout_rate_l{}".format(i), self.cnn_hyp_par['dropout_min'], self.cnn_hyp_par['dropout_max'])
          model.add(Dropout(dropout_rate))
          model.add(BatchNormalization())
      model.add(Dense(self.num_classes, activation="softmax"))
      learning_rate = trial.suggest_float("learning_rate", self.cnn_hyp_par['learning_rate_min'], self.cnn_hyp_par['learning_rate_max'], log=True)
      model.compile(
          loss="sparse_categorical_crossentropy",
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Use tf.keras.optimizers
          metrics=["accuracy"],
      )
      early_stop = EarlyStopping(monitor='val_loss', patience=3)
      #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
      model.fit(
            X_train, self.Y_train, validation_data=(X_valid, self.Y_valid),
            shuffle=True, batch_size=self.batch_size,
            epochs=self.epochs, verbose=False, callbacks=[early_stop]
        )
      score = model.evaluate(X_valid, self.Y_valid, verbose=0)
      print(f"Trial {trial.number}, Score: {score[1]}")
      print(f"Parameters: {trial.params}")
      return score[1]

    def create_transformer_model(self, input_dim, num_classes, d_model, num_heads, num_layers, dropout, weight_decay, l1_regularization):
        """"
        This method takes the hyperparameters of the transformer, creates a model and then trains and tests the model 
        """
      d_model = int(np.ceil(d_model / num_heads) * num_heads)
      class TabularTransformer(nn.Module):
        def __init__(self):
          super(TabularTransformer, self).__init__()
          self.embedding = nn.Linear(input_dim, d_model)
          self.embedding_dropout = nn.Dropout(dropout)
          self.transformer_layers = nn.ModuleList([
              nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=2 * d_model, dropout=dropout)
              for _ in range(num_layers)
          ])
          self.classifier = nn.Linear(d_model, num_classes)
          self.weight_decay = weight_decay
          self.l1_regularization = l1_regularization
        def forward(self, x):
          x = self.embedding(x)
          x = self.embedding_dropout(x)
          for layer in self.transformer_layers:
              x = layer(x)
          x = self.classifier(x)
          return F.log_softmax(x, dim=1), self.l1_regularization * torch.norm(self.embedding.weight.data, 1)
      return TabularTransformer()

    def transformer_objective(self, trial):
        num_workers = 2
        encoder = LabelEncoder()
        Y_train_encoded = encoder.fit_transform(self.Y_train)
        y_train_tensor = torch.tensor(Y_train_encoded, dtype=torch.int64)
        scaler = StandardScaler()
        X_train_new = scaler.fit_transform(self.X_train)
        X_train_tensor = torch.tensor(X_train_new, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        X_valid_new = scaler.transform(self.X_valid)
        X_valid_tensor = torch.tensor(X_valid_new, dtype=torch.float32)
        y_valid_encoded = encoder.transform(self.Y_valid)
        y_valid_tensor = torch.tensor(y_valid_encoded, dtype=torch.int64)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=num_workers)
        d_model = trial.suggest_int('d_model', *self.transformer_hyp_par['d_model'])
        num_heads = trial.suggest_int('num_heads', *self.transformer_hyp_par['num_heads'])
        num_layers = trial.suggest_int('num_layers', *self.transformer_hyp_par['num_layers'])
        dropout = trial.suggest_float('dropout', *self.transformer_hyp_par['dropout'])
        weight_decay = trial.suggest_float('weight_decay', *self.transformer_hyp_par['weight_decay'], log=True)
        l1_regularization = trial.suggest_float('l1_regularization', *self.transformer_hyp_par['l1_regularization'], log=True)
        learning_rate = trial.suggest_float('learning_rate', *self.transformer_hyp_par['learning_rate'], log=True)
        
        if d_model % num_heads != 0:
          return 1000.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.create_transformer_model(self.input_shape[0], len(np.unique(self.y_train)), d_model, num_heads, num_layers, dropout, weight_decay, l1_regularization).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        scaler = amp.GradScaler()
        
        best_validation_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(10):
          running_loss = 0.0
          for i, data in enumerate(train_loader):
              inputs, labels = data[0].to(device), data[1].to(device)
              optimizer.zero_grad()
              with amp.autocast():
                  outputs, l1_loss = model(inputs)
                  loss = criterion(outputs, labels) + l1_loss
        
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()
        
              running_loss += loss.item()
        
          validation_loss = 0.0
          with torch.no_grad():
              for data in valid_loader:
                  inputs, labels = data[0].to(device), data[1].to(device)
                  outputs, _ = model(inputs)
                  loss = criterion(outputs, labels)
                  validation_loss += loss.item()
          scheduler.step(validation_loss)
          if validation_loss < best_validation_loss:
              best_validation_loss = validation_loss
              epochs_without_improvement = 0
          else:
              epochs_without_improvement += 1
          if epochs_without_improvement > 10:
              break
        return validation_loss

    def optimize_transformer(self):
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                    sampler=optuna.samplers.TPESampler())

        study.optimize(lambda trial: self.transformer_objective(trial), n_trials=self.n_trials, gc_after_trial=True)
        print("Transformer Best Trial:")
        print(study.best_trial.params)
        return study

    def optimize(self, model_type):
      study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(), sampler=optuna.samplers.TPESampler())

      if model_type == "MLP":
          study.optimize(self.MLP_objective, n_trials=self.n_trials, gc_after_trial=True)
          print("MLP Best Trial:")
      elif model_type == "xgboost":
          study.optimize(self.xgboost_objective, n_trials=self.n_trials, gc_after_trial=True)
          print("XGBoost Best Trial:")
      elif model_type == "cnn":
          study.optimize(self.cnn_objective, n_trials=self.n_trials, gc_after_trial=True)
          print("CNN Best Trial:")
      elif model_type == "transformer":
          if self.transformer_hyp_par is None:
              raise ValueError("Hyperparameters for Transformer model are missing.")
          study.optimize(lambda trial: self.transformer_objective(trial), n_trials=self.n_trials, gc_after_trial=True)
          print("Transformer Best Trial:")
      else:
          raise ValueError("Invalid model_type. Choose 'MLP', 'xgboost', 'cnn', or 'transformer'.")
      return study
