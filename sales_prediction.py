"""
CodeAlpha Task: Sales Prediction using Python
---------------------------------------------
Predicts sales from advertising spend on TV, Radio and Newspaper, compares
regression models, measures how each channel moves sales and turns the
findings into marketing recommendations.

Units (as in the dataset): ad spend = thousands of dollars,
sales = thousands of units.

Run:  python sales_prediction.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
FIG = BASE / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
SEED = 42
CHANNELS = ["TV", "Radio", "Newspaper"]


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.close()


# ------------------------------------------------------------ 1. LOAD & CLEAN
def load_and_clean():
    df = pd.read_csv(BASE / "data" / "Advertising.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])  # row-id column
    df.columns = df.columns.str.strip()
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ------------------------------------------------------------ 2. FEATURES
def engineer(df):
    df = df.copy()
    df["TV_x_Radio"] = df["TV"] * df["Radio"]          # TV and radio may reinforce each other
    df["Total_Spend"] = df[CHANNELS].sum(axis=1)
    return df


# ------------------------------------------------------------ 3. EDA
def explore(df):
    print("\nSummary statistics:\n", df[CHANNELS + ["Sales"]].describe().round(2))
    print("\nCorrelation with Sales:\n", df[CHANNELS + ["Sales"]].corr()["Sales"].drop("Sales").round(3).to_string())

    sns.pairplot(df[CHANNELS + ["Sales"]], kind="reg", plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.5}})
    plt.suptitle("Advertising channels vs sales", y=1.02)
    save("01_pairplot")

    sns.heatmap(df[CHANNELS + ["Sales"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    save("02_correlation")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    for a, c in zip(ax, CHANNELS + ["Sales"]):
        sns.histplot(df[c], kde=True, ax=a, color="steelblue")
    save("03_distributions")


# ------------------------------------------------------------ 4. MODELS
def evaluate(df):
    sets = {
        "Linear (3 channels)": CHANNELS,
        "Linear + TV x Radio": CHANNELS + ["TV_x_Radio"],
        "Linear (TV + Radio only)": ["TV", "Radio"],
        "Linear + TV x Radio (TV, Radio only)": ["TV", "Radio", "TV_x_Radio"],
    }
    models = {}
    for n, cols in sets.items():
        models[n] = (make_pipeline(StandardScaler(), LinearRegression()), cols)
    models["Ridge (3 channels + interaction)"] = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), CHANNELS + ["TV_x_Radio"])
    models["Random Forest"] = (RandomForestRegressor(n_estimators=300, random_state=SEED), CHANNELS)
    models["Gradient Boosting"] = (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                                            random_state=SEED), CHANNELS)

    y = df["Sales"]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=SEED)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows, fitted = [], {}
    for name, (m, cols) in models.items():
        cvr = cross_validate(m, df[cols], y, cv=cv, scoring=scoring)
        m.fit(df.loc[tr_idx, cols], y[tr_idx])
        pred = m.predict(df.loc[te_idx, cols])
        rows.append({
            "Model": name,
            "CV R2": cvr["test_r2"].mean(), "CV R2 std": cvr["test_r2"].std(),
            "CV MAE": -cvr["test_mae"].mean(), "CV RMSE": -cvr["test_rmse"].mean(),
            "Test R2": r2_score(y[te_idx], pred), "Test MAE": mean_absolute_error(y[te_idx], pred),
        })
        fitted[name] = (m, cols)
    res = pd.DataFrame(rows).set_index("Model").round(3).sort_values("CV R2", ascending=False)
    print("\nMODEL COMPARISON (sorted by cross-validated R2; MAE/RMSE in thousand units)\n", res.to_string())
    res.to_csv(BASE / "outputs" / "model_comparison.csv")

    plt.figure(figsize=(9, 4.5))
    res[["CV R2", "Test R2"]].sort_values("CV R2").plot(kind="barh", figsize=(9, 4.5))
    plt.xlim(0.8, 1.0); plt.xlabel("R2 (axis starts at 0.8)"); plt.title("Model comparison")
    save("04_model_comparison")
    return res, fitted, tr_idx, te_idx


# ------------------------------------------------------------ 5. BEST MODEL
def report_best(df, res, fitted, te_idx):
    best = res.index[0]
    model, cols = fitted[best]
    y_te = df.loc[te_idx, "Sales"]
    pred = model.predict(df.loc[te_idx, cols])
    print(f"\nBest model: {best} | CV R2 = {res.loc[best, 'CV R2']:.3f} | Test R2 = {res.loc[best, 'Test R2']:.3f}"
          f" | Test MAE = {res.loc[best, 'Test MAE']:.2f} thousand units")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, alpha=0.8, color="navy")
    lim = [y_te.min() - 1, y_te.max() + 1]
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual sales"); plt.ylabel("Predicted sales"); plt.legend()
    plt.title(f"Actual vs predicted - {best}")
    save("05_actual_vs_predicted")

    resid = y_te - pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(pred, resid, alpha=0.8); ax[0].axhline(0, color="red", ls="--")
    ax[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
    sns.histplot(resid, kde=True, ax=ax[1]); ax[1].set_title("Residual distribution")
    save("06_residuals")

    joblib.dump({"model": model, "features": cols}, BASE / "outputs" / "best_model.joblib")
    return best


# ------------------------------------------------------------ 6. ADVERTISING IMPACT
def impact(df):
    """Effect of each channel, using a plain linear model on the original units."""
    print("\n" + "=" * 70, "\nHOW ADVERTISING AFFECTS SALES\n", "=" * 70, sep="")
    X, y = df[CHANNELS], df["Sales"]
    lin = LinearRegression().fit(X, y)
    # Bootstrap 95% intervals: how stable is each channel's effect?
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(df), len(df))
        boots.append(LinearRegression().fit(X.iloc[i], y.iloc[i]).coef_)
    boots = np.array(boots)
    tab = pd.DataFrame({
        "Effect on sales per +1 (thousand $)": lin.coef_,
        "95% CI low": np.percentile(boots, 2.5, axis=0),
        "95% CI high": np.percentile(boots, 97.5, axis=0),
    }, index=CHANNELS).round(3)
    print(f"\nBaseline sales with zero advertising (intercept): {lin.intercept_:.2f} thousand units")
    print("\nEach extra $1,000 of spend in one channel (others held constant) adds this many thousand units:")
    print(tab.to_string())
    print("\nA 95% interval that includes 0 means the channel has no reliable effect once TV and radio are accounted for.")

    plt.figure(figsize=(6, 4))
    plt.errorbar(CHANNELS, tab.iloc[:, 0],
                 yerr=[tab.iloc[:, 0] - tab["95% CI low"], tab["95% CI high"] - tab.iloc[:, 0]],
                 fmt="o", capsize=6, color="darkred")
    plt.axhline(0, color="grey", ls="--")
    plt.ylabel("Added sales per +$1k spend (thousand units)")
    plt.title("Channel effect with 95% confidence interval")
    save("07_channel_effects")

    # Interaction model: does combining TV and radio help?
    inter = LinearRegression().fit(df[["TV", "Radio", "TV_x_Radio"]], y)
    return lin, inter


def scenarios(df, inter):
    print("\n" + "=" * 70, "\nBUDGET SCENARIOS (TV + Radio interaction model)\n", "=" * 70, sep="")

    def predict(tv, radio):
        return inter.predict(pd.DataFrame({"TV": [tv], "Radio": [radio], "TV_x_Radio": [tv * radio]}))[0]

    tv0, ra0 = df["TV"].mean(), df["Radio"].mean()
    base = predict(tv0, ra0)
    print(f"\nAverage budget (TV {tv0:.0f}k, Radio {ra0:.0f}k) -> predicted sales {base:.2f}k units")
    for label, tv, ra in [
        ("+$10k on TV only", tv0 + 10, ra0),
        ("+$10k on Radio only", tv0, ra0 + 10),
        ("+$5k on each of TV and Radio", tv0 + 5, ra0 + 5),
    ]:
        p = predict(tv, ra)
        print(f"  {label:32s} -> {p:.2f}k units ({p - base:+.2f}k)")

    # Same total budget split different ways, restricted to spends seen in the data
    total = tv0 + ra0
    tv_max, ra_max = df["TV"].max(), df["Radio"].max()
    splits = np.array([s for s in np.linspace(0, 1, 101)
                       if total * (1 - s) <= tv_max and total * s <= ra_max])
    sales = [predict(total * (1 - s), total * s) for s in splits]
    k = int(np.argmax(sales))
    print(f"\nSame total TV+Radio budget ({total:.0f}k), splits kept inside the observed spend range "
          f"(radio <= {ra_max:.0f}k): best is {(1 - splits[k]) * 100:.0f}% TV / {splits[k] * 100:.0f}% Radio "
          f"-> {sales[k]:.2f}k units")
    print("The best split sits at the radio limit seen in the data, so the model says 'more radio' "
          "but cannot say how far that holds beyond ~50k.")

    plt.figure(figsize=(7, 4))
    plt.plot(splits * 100, sales, color="teal")
    plt.axvline(splits[k] * 100, color="red", ls="--", label=f"Best in range: {splits[k] * 100:.0f}% radio")
    plt.xlabel("Share of TV+Radio budget put into Radio (%)"); plt.ylabel("Predicted sales (thousand units)")
    plt.title("Same budget, different TV/Radio split (within observed range)"); plt.legend()
    save("08_budget_split")

    # Response curves
    grid = np.linspace(0, df["TV"].max(), 50)
    plt.figure(figsize=(7, 4))
    for r in (0, 25, 50):
        plt.plot(grid, [predict(t, r) for t in grid], label=f"Radio = {r}k")
    plt.xlabel("TV spend (thousand $)"); plt.ylabel("Predicted sales")
    plt.title("TV response curve at different radio budgets"); plt.legend()
    save("09_tv_response_by_radio")


def insights():
    print("\n" + "=" * 70, "\nMARKETING RECOMMENDATIONS\n", "=" * 70, sep="")
    print("""
1. TV and Radio drive sales; Newspaper adds little or nothing once they are
   accounted for (its raw correlation with sales comes from being correlated
   with Radio spend). Review or cut newspaper spend.
2. Each extra $1k on Radio brings a larger lift than $1k on TV, so Radio is the
   more cost-efficient channel at the margin.
3. TV and Radio reinforce each other (positive interaction): a mixed budget
   beats putting everything in one channel.
4. Shift money from Newspaper toward Radio/TV and track the result.
5. Limits: only 200 observations, one snapshot (no dates, so no time-series
   forecast), and no segment/platform data. Test any big budget change in a
   small experiment before scaling it.
""")


def main():
    df = engineer(load_and_clean())
    explore(df)
    res, fitted, tr_idx, te_idx = evaluate(df)
    report_best(df, res, fitted, te_idx)
    lin, inter = impact(df)
    scenarios(df, inter)
    insights()
    print(f"Figures saved in {FIG}")


if __name__ == "__main__":
    main()
