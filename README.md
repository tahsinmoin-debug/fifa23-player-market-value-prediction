# FIFA 23 Player Market Value Prediction

Predicting a football player's market value (in €) from their FIFA 23 attributes using regression models. Built as a group project for **CSE427 (Machine Learning), BRAC University**.

## Dataset

- **Source:** [FIFA 23 Players Data](https://www.kaggle.com/datasets/sanjeetsinghnaik/fifa-23-players-dataset) (Kaggle), included in [`data/`](data/)
- **Size:** ~18,500 players × 89 columns
- **Target:** `Value(in Euro)`
- **Features:** overall rating, potential, age, height/weight, contract details, skill moves, weak foot, international reputation, work rates, and 30+ technical attributes (pace, shooting, passing, dribbling, defending, physicality and their sub-stats)

## Workflow

1. **EDA:** rating distribution, correlation heatmap, market value vs overall by age group, preferred foot / skill moves, age vs potential
2. **Cleaning:** dropped duplicates and identifier columns (names, image links), parsed currency strings (`€`, `,`) to numbers, converted `Joined On` to year, imputed missing values (median / mode), label-encoded categoricals
3. **Target processing:** removed zero-value players and applied a `log1p` transform to fix heavy right skew
4. **Leakage control:** dropped `Release Clause` (it's derived directly from value)
5. **Modelling:** 80/20 train-test split, `StandardScaler` for linear models and MLP, raw features for tree models
6. **Evaluation:** RMSE, MAE and R² on the log-scaled target

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Extra Trees** | **0.0520** | **0.0310** | **0.9981** |
| XGBoost | 0.0547 | 0.0391 | 0.9979 |
| Random Forest | 0.0554 | 0.0331 | 0.9979 |
| Gradient Boosting | 0.0763 | 0.0584 | 0.9960 |
| Decision Tree | 0.0970 | 0.0435 | 0.9935 |
| Ridge Regression | 0.2059 | 0.1524 | 0.9707 |
| Linear Regression | 0.2060 | 0.1526 | 0.9707 |
| Lasso Regression | 0.2085 | 0.1533 | 0.9700 |
| MLP | 0.2389 | 0.1504 | 0.9606 |

Tree-based ensembles clearly beat the linear models, because the link between attributes and value is strongly non-linear (value grows sharply at high ratings and drops with age).

### A note on these scores

R² ≈ 0.998 is very high. Part of the reason is that the original run kept `Wage(in Euro)` as a feature, and wage is closely tied to market value in FIFA's data. A stricter version would drop wage too and predict value only from skill and profile attributes. That's the next step listed below.

## Repo structure

```
├── data/
│   └── Fifa 23 Players Data.csv                 # raw dataset
├── notebooks/
│   ├── fifa23_market_value_prediction.ipynb        # original Colab notebook (with outputs)
│   └── fifa23_market_value_prediction_clean.ipynb  # refactored: no Drive dependency, adds XGBoost & MLP
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fifa23_market_value_prediction_clean.ipynb
```

The clean notebook finds `data/Fifa 23 Players Data.csv` automatically. The original notebook was written for Colab and reads from Google Drive.

## Next steps

- Retrain without `Wage` to measure performance from attributes alone
- Hyperparameter tuning with cross-validation
- SHAP values to explain which attributes drive value
- Report errors in € (inverse of the log transform) so they're easier to read
