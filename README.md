# 🚀 Crypto-eKAN

**Crypto-eKAN** is an intelligent ensemble forecasting system for cryptocurrency price prediction powered by machine learning, deep learning, Monte Carlo simulation, and a stacked **Kernel Activation Network (KAN)**. This repository blends classical models with neural architectures into a meta-learning pipeline designed to capture market dynamics and forecast asset prices like XRP and BTC.

---

## 📌 Features

- **Modular Architecture** — clean separation of components like preprocessing, modeling, stacking, and evaluation  
- **Stacked Ensemble** — combines predictions from GRU, LSTM, and MLP using a PyTorch-powered KAN meta-learner  
- **Monte Carlo Simulation** — includes GBM, CIR, and Normal methods for single-day crypto price forecasting  
- **Model Comparison** — integrates [NeuralForecast](https://github.com/Nixtla/neuralforecast) and [StatsForecast](https://github.com/Nixtla/statsforecast) tools  
- **Visual Diagnostics** — actual vs predicted scatter plots, residual plots, histograms, sensitivity maps, and learning curves  
- **Optuna Integration** — automated hyperparameter tuning across models  

---

## 🧠 Architecture Overview

| Module               | Algorithms Included                        | Description                                |
|---------------------|--------------------------------------------|--------------------------------------------|
| `data_pipeline.py`   | Lag features, SMA, EMA, RSI                | Crypto data loading & feature engineering  |
| `models_ml.py`       | RF, GBR, Ridge/Lasso                       | Classical regressors                        |
| `models_dl.py`       | MLP, LSTM, GRU                             | Deep learning via Keras                     |
| `ensemble_kan.py`    | PyTorch-based KAN meta-learner             | Stacks outputs of DL models                |
| `monte_carlo.py`     | GBM, CIR, Normal simulations               | Stochastic price prediction methods         |
| `evaluation_tools.py`| Metrics + plots                            | Evaluation and visual analytics             |

---

## 🧪 Quick Start

### 1. Clone Repo & Install

```bash
git clone https://github.com/yourusername/Crypto-eKAN.git
cd Crypto-eKAN
pip install -r requirements.txt
```

### 2. Load & Process Data

```bash
from data_pipeline import download_data, engineer_features, split_data, create_xy

df = download_data('BTC-USD')
df, scaler = engineer_features(df)
train_df, val_df, test_df = split_data(df)
X_train, y_train = create_xy(train_df)
X_val, y_val = create_xy(val_df)
X_test, y_test = create_xy(test_df)
```

### 3. Train a Model (Example: MLP)

```bash
from models_dl import bMLP
model = bMLP(best_params, input_shape=X_train.shape[1])
model.fit(X_train, y_train, epochs=50, ...)
```

### 4. Ensemble via KAN

```bash
from ensemble_kan import train_stacked_model, prepare_meta_inputs
val_preds = [val_preds_lstm, val_preds_gru, val_preds_mlp]
test_preds = [test_preds_lstm, test_preds_gru, test_preds_mlp]

X_meta_train, y_meta_train, X_meta_test, y_test_tensor = prepare_meta_inputs(val_preds, test_preds, y_val, y_test)
model, preds, r2_score_final = train_stacked_model(X_meta_train, y_meta_train, X_meta_test, y_test_tensor)
```

### 5. Monte Carlo Prediction

```bash
from monte_carlo import monte_carlo_next_day_gbm
next_price = monte_carlo_next_day_gbm(prices)
```

## 📊 Model Performance
- High accuracy from KAN meta-learning (R² > 0.9 in tuned cases)
- Monte Carlo methods effective for volatility adjustment
- GRU and LSTM outperform classical models in long horizons
- Learning curves show stable training across ensembles

## 🧱 Contributing
Want to add Transformers, multi-horizon models, or regime detection? Feel free to fork, star ⭐, and propose ideas via pull requests or issues!

## 📜 License
MIT License © 2025 — Use freely with attribution
