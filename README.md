# NYC Yellow Taxi Fare Prediction (Linear Regression)

Predicts taxi `fare_amount` for 2017 NYC Yellow Taxi trips using multiple linear regression in scikit-learn. The notebook covers the full workflow: data cleaning, outlier handling, feature engineering, modeling, and evaluation.

## Project structure

```
Linear_Regression_Using_Scikit_Learn/
├── taxi_fare_regression.ipynb      # main notebook
├── 2017_Yellow_Taxi_Trip_Data.csv  # dataset (place here)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Dataset

`2017_Yellow_Taxi_Trip_Data.csv` contains ~23,000 yellow taxi trips from 2017 (NYC TLC trip records). Key columns:

| Column | Description |
|---|---|
| `tpep_pickup_datetime`, `tpep_dropoff_datetime` | Trip start and end timestamps |
| `trip_distance` | Distance in miles |
| `PULocationID`, `DOLocationID` | Pickup and dropoff zone IDs |
| `RatecodeID` | Rate type (2 = JFK flat fare) |
| `fare_amount` | Metered fare in USD (**target**) |
| `VendorID`, `passenger_count` | Vendor and passenger count |

## Workflow

1. **Load and inspect:** shape, dtypes, duplicates, missing values.
2. **Missing-value visualization:** matrix, per-column non-null counts, and nullity correlation using matplotlib/seaborn.
3. **Datetime handling:** parse pickup/dropoff columns and compute `duration` in minutes.
4. **Outlier handling:**
   - Boxplots for `trip_distance`, `fare_amount`, `duration`.
   - Negative fares set to 0.
   - `fare_amount` and `duration` capped at `Q3 + 6 * IQR`.
   - `trip_distance` left unchanged (short trips are real, not errors).
5. **Feature engineering:**
   - `mean_distance`: average distance per pickup/dropoff zone pair.
   - `mean_duration`: average duration per pickup/dropoff zone pair.
   - `day`, `month`, and `rush_hour` (weekdays 06:00-10:00 and 16:00-20:00).
6. **Exploration:** scatter plot, pair plot, correlation matrix and heatmap.
7. **Modeling:** one-hot encode `VendorID`, 80/20 train/test split (`random_state=1`), `StandardScaler`, `LinearRegression`.
8. **Evaluation:** R², MAE, MSE, RMSE on train and test sets, plus actual-vs-predicted and residual plots.

## Key findings

- `mean_distance` (r = 0.91) and `mean_duration` (r = 0.86) are the strongest predictors of fare.
- `VendorID`, `passenger_count`, and `rush_hour` have near-zero linear correlation with fare.
- After standardization, `mean_distance` has the largest coefficient (~7.17), then `mean_duration` (~2.77).
- The model reaches an R² of about 0.84 on the training data.
- Trips with `RatecodeID` 2 (JFK) have a regulated $52 flat fare. Those fares can be set directly after prediction instead of relying on the model.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv add --dev ipykernel
```

If you are creating the project from scratch:

```powershell
uv add pandas numpy matplotlib seaborn scikit-learn statsmodels
uv add --dev ipykernel
```

## Run

1. Place `2017_Yellow_Taxi_Trip_Data.csv` in the same folder as the notebook.
2. Open `taxi_fare_regression.ipynb` in VS Code or Jupyter.
3. Select the project's `.venv` as the kernel.
4. Run all cells from the top.

The notebook reads the CSV from the current working directory (`Path.cwd()`).

## Troubleshooting

- **`No module named 'matplotlib_inline'` or `'IPython.core...'`:** the environment has partially installed packages. Reinstall them:
  ```powershell
  uv sync --reinstall-package ipython --reinstall-package ipykernel --reinstall-package matplotlib-inline --reinstall-package pygments
  ```
  If that fails, delete `.venv` and run `uv sync` again.
- **`FileNotFoundError` for the CSV:** the file is not in the notebook's working directory.

## Possible improvements

- Fix `outlier_imputer` so it returns a useful value instead of printing `None`.
- Try tree-based models (Random Forest, XGBoost) and compare with the linear baseline.
- Use cross-validation instead of a single train/test split.
- Apply the JFK flat-fare rule to final predictions.

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels, uv