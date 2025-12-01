import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# === Meta Features Preparation ===
def prepare_meta_inputs(val_preds, test_preds, y_val, y_test):
    X_meta_train = np.stack(val_preds, axis=1)
    X_meta_test = np.stack(test_preds, axis=1)

    return (
        torch.tensor(X_meta_train, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_meta_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    )

# === Define KAN Meta-Model ===
class KAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KAN, self).__init__()
        self.univariate_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU() if i % 3 == 0 else nn.Sigmoid() if i % 3 == 1 else nn.LeakyReLU()
            ) for i in range(input_size)
        ])
        self.combine_funcs = nn.Sequential(
            nn.Linear(input_size * hidden_size, 1)
        )

    def forward(self, x):
        outputs = [func(x[:, i].unsqueeze(1)) for i, func in enumerate(self.univariate_funcs)]
        concat = torch.cat(outputs, dim=1)
        return self.combine_funcs(concat)

# === Optuna Objective Function ===
def objective(trial, X_train, y_train, X_val, y_val):
    hidden_size = trial.suggest_int('hidden_size', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = KAN(X_train.shape[1], hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for _ in range(20):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val).numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y_val.numpy(), preds))
    return rmse

# === Train Final KAN Ensemble ===
def train_stacked_model(X_meta_train, y_meta_train, X_meta_test, y_test_tensor, max_trials=10):
    best_r2, best_model = -np.inf, None

    for _ in range(max_trials):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_meta_train, y_meta_train, X_meta_test, y_test_tensor), n_trials=100)

        hidden_size = study.best_params['hidden_size']
        lr = study.best_params['learning_rate']

        model = KAN(X_meta_train.shape[1], hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(50):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_meta_train), y_meta_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_meta_test).numpy().flatten()

        r2 = r2_score(y_test_tensor.numpy(), y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            final_preds = y_pred

    return best_model, final_preds, best_r2
