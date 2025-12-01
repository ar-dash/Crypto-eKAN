import numpy as np
import optuna
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, GRU, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 🧮 Performance Metrics
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2}

# 🔬 MLP Model Builder
def bMLP(best_params, input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=(input_shape,)))
    for _ in range(best_params['n_layers']):
        model.add(Dense(units=best_params['n_units'], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    return model

# 🌊 LSTM Model Builder
def bLSTM(best_params, input_shape):
    model = Sequential()
    model.add(LSTM(units=best_params['n_units'], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(LSTM(units=best_params['n_units']))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    return model

# 🌀 GRU Model Builder
def bGRU(best_params, input_shape):
    model = Sequential()
    model.add(GRU(units=best_params['n_units'], return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=best_params['n_units']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    return model

# 🔍 Optuna Objective for MLP
def objective_mlp(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'n_units': trial.suggest_int('n_units', 50, 200),
        'n_epochs': trial.suggest_int('n_epochs', 10, 100),
        'n_batch': trial.suggest_categorical('n_batch', [16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }
    model = bMLP(params, X_train.shape[1])
    model.fit(X_train, y_train, epochs=params['n_epochs'], batch_size=params['n_batch'], validation_data=(X_val, y_val), verbose=0)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# 🔍 Optuna Objective for LSTM
def objective_lstm(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_units': trial.suggest_int('n_units', 10, 100),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'n_batch': trial.suggest_int('n_batch', 16, 128)
    }
    model = bLSTM(params, input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=params['n_batch'], validation_data=(X_val, y_val), verbose=0)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# 🔍 Optuna Objective for GRU
def objective_gru(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_units': trial.suggest_int('n_units', 10, 200),
        'n_epochs': trial.suggest_int('n_epochs', 10, 100),
        'n_batch': trial.suggest_categorical('n_batch', [16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }
    model = bGRU(params, input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=params['n_epochs'], batch_size=params['n_batch'], validation_data=(X_val, y_val), verbose=0)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))
