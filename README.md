# CodeAlpha – Sales Prediction using Python

Predicts sales from advertising spend on TV, Radio and Newspaper, measures how each channel affects sales, and turns the results into marketing recommendations.

## Structure
```
data/Advertising.csv
sales_prediction.py
requirements.txt
outputs/   analysis_output.txt, model_comparison.csv, best_model.joblib, figures/ (9 charts)
```

## Run
```
pip install -r requirements.txt
python sales_prediction.py
```

## Dataset
200 rows: `TV, Radio, Newspaper` (ad spend, thousand $) and `Sales` (thousand units). No missing values or duplicates; the row-id column was dropped. The data has no dates, so regression is used (not time series), and "platform" means the three ad channels. There is no target-segment column.

## Workflow
1. **Cleaning:** removed the id column, checked missing values and duplicates.
2. **Feature engineering:** `TV_x_Radio` (interaction) and `Total_Spend`.
3. **Models:** Linear regression variants, Ridge, Random Forest, Gradient Boosting.
4. **Evaluation:** repeated 5-fold cross-validation (5 repeats) plus a held-out 20% test set; R², MAE, RMSE.
5. **Impact analysis:** channel coefficients with bootstrap 95% intervals, budget scenarios, response curves.

## Results
| Model | CV R² | CV MAE (thousand units) |
|---|---|---|
| Random Forest | 0.978 | 0.57 |
| Gradient Boosting | 0.975 | 0.57 |
| Linear + TV×Radio | 0.964 | 0.69 |
| Linear (3 channels) | 0.881 | 1.29 |

Adding the TV×Radio interaction lifts a plain linear model from R² 0.88 to 0.96, so TV and Radio reinforce each other.

## Effect of +$1,000 spend (others constant)
| Channel | Added sales (thousand units) | 95% CI |
|---|---|---|
| TV | 0.046 | 0.042 to 0.049 |
| Radio | 0.189 | 0.168 to 0.209 |
| Newspaper | -0.001 | -0.012 to 0.012 |

## Recommendations
- Newspaper shows no reliable effect once TV and Radio are included; review that spend.
- Radio gives the biggest lift per extra dollar; TV also matters.
- TV and Radio work better together than alone, so keep a mixed budget.
- Test budget shifts in a small experiment first.

## Limitations
Only 200 observations, one snapshot in time, and results are only reliable within the spend ranges in the data (e.g. radio up to ~50k). Correlation from observational data does not prove cause.
