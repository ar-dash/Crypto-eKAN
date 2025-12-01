import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# 📊 Performance Metrics
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2}

# 🌲 Random Forest Regressor with pre-optimized params
def bRFR(best_params):
    return RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

# 🔁 Ridge/Lasso Regression
def bLR(best_params):
    if best_params["model_type"] == "ridge":
        return Ridge(alpha=best_params["alpha"], random_state=42)
    else:
        return Lasso(alpha=best_params["alpha"], random_state=42)

# 🌿 Gradient Boosting Regressor
def bGBR(best_params):
    return GradientBoostingRegressor(**best_params, random_state=42)

# 🔍 Optuna Objective for Ridge vs Lasso
def objective_lr(trial, X_train, y_train, X_val, y_val):
    model_type = trial.suggest_categorical("model_type", ["ridge", "lasso"])
    alpha = trial.suggest_float("alpha", 0.0001, 10.0, log=True)

    model = Ridge(alpha=alpha) if model_type == "ridge" else Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# 🔍 Optuna Objective for Gradient Boosting
def objective_gbr(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))
