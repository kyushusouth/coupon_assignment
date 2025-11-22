import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

sns.set_style("whitegrid")

random_seed = 42
n_users = 1000000
coupon_settings = {"A": 1000, "B": 1500}
coupon_types = list(coupon_settings.keys())


def create_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    base_cols = {
        "user_id": np.arange(n_users),
        "age": rng.integers(18, 71, n_users),
        "gender_code": rng.choice([0, 1], n_users, replace=True, p=[0.5, 0.5]),
        "past_purchases": rng.poisson(3, n_users),
        "avg_spend": np.clip(rng.normal(5000, 1500, n_users), 0, None).astype(int),
    }

    df = pd.DataFrame(base_cols)
    df["coupon"] = rng.choice(coupon_types, n_users)
    df["discount_amount"] = df["coupon"].map(coupon_settings)

    beta_intercept = -10
    beta_age = -0.002  # 年齢が低い方がcvしやすい
    beta_gender = 0.0001  # gender_code=1の方がcvしやすい
    beta_past_purchases = 0.02  # 過去購入回数が多い方がcvしやすい
    beta_avg_spend = 0.0001  # 過去平均支出額が高い方がcvしやすい

    user_sensitivity = (
        0.001  # ベースラインの感応度
        + (df["age"] - 45) * 0.00005  # 年齢が高いほどプラス
        - (df["avg_spend"] - 5000)
        * 0.000001  # 平均支出が高いとマイナス（あまり響かない）
        + (df["past_purchases"] < 2) * 0.0005  # 過去購入が少ない人は反応しやすい
    )
    base_discount_effect = 1000 * 0.001
    tau_effect = (df["discount_amount"] - coupon_settings["A"]) * user_sensitivity

    noise = rng.normal(0, 0.3, n_users)

    logit = (
        beta_intercept
        + df["age"] * beta_age
        + (df["gender_code"] == 1) * beta_gender
        + df["past_purchases"] * beta_past_purchases
        + df["avg_spend"] * beta_avg_spend
        + base_discount_effect
        + tau_effect
        + noise
    )
    cvr = 1 / (1 + np.exp(-logit))
    df["cv"] = rng.binomial(1, cvr)

    ordinal_encoder = OrdinalEncoder(
        categories=[coupon_types],
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    df["coupon_encoded"] = ordinal_encoder.fit_transform(df[["coupon"]])
    return df


def assign_coupon_by_score(
    df_test_score: pd.DataFrame, df_test: pd.DataFrame, cvr_name: str, score_name: str
) -> pd.DataFrame:
    # 制限なし
    coupon_budgets = {
        "A": len(df_test),
        "B": len(df_test),
    }

    df_test_score[f"{score_name}_B-A"] = (
        df_test_score[f"{cvr_name}_B"] - df_test_score[f"{cvr_name}_A"]
    ) / (coupon_settings["B"] - coupon_settings["A"])

    df_test_score = pd.melt(
        df_test_score,
        id_vars=["user_id"],
        value_vars=[f"{score_name}_B-A"],
        var_name="score_type",
        value_name="score",
    )

    df_test_score = df_test_score.sort_values("score", ascending=False)
    assigned_users = set()
    for row in tqdm(
        df_test_score.itertuples(), desc="Assigning coupons", total=len(df_test_score)
    ):
        user_id = row.user_id
        if user_id in assigned_users:
            continue

        coupon = row.score_type.split("-")[0].split("_")[2]
        if coupon_budgets[coupon] > 0:
            df_test.loc[df_test["user_id"] == user_id, f"{score_name}_coupon"] = coupon
            df_test.loc[df_test["user_id"] == user_id, score_name] = row.score
            coupon_budgets[coupon] -= 1
            assigned_users.add(user_id)
        else:
            continue

    df_test[f"{score_name}_coupon"] = df_test[f"{score_name}_coupon"].fillna("A")
    return df_test


def calc_cumulative_uplift(
    df_test: pd.DataFrame,
    df_uplift: pd.DataFrame,
    score_name: str,
    cum_cvr_uplift_name: str,
    cum_cv_uplift_name: str,
) -> pd.DataFrame:
    df_test = df_test.sort_values(score_name, ascending=False)

    model_cv = df_test.apply(
        lambda row: row["cv"]
        if row["coupon"] == row[f"{score_name}_coupon"]
        else np.nan,
        axis=1,
    )
    baseline_cv = df_test.apply(
        lambda row: row["cv"] if row["coupon"] == "A" else np.nan,
        axis=1,
    )

    cum_model_cvr = model_cv.expanding().mean()
    cum_baseline_cvr = baseline_cv.expanding().mean()
    cum_cvr_uplift = cum_model_cvr - cum_baseline_cvr

    df_uplift[cum_cvr_uplift_name] = cum_cvr_uplift.to_numpy()
    df_uplift[cum_cv_uplift_name] = (
        df_uplift[cum_cvr_uplift_name] * df_uplift["population"]
    )
    return df_uplift


def main():
    root_dir = Path("~/dev/coupon_assignment").expanduser()
    result_dir = (
        root_dir
        / "result"
        / datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9))
        ).strftime("%Y%m%d_%H%M%S")
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    df = create_dummy_data()
    print(df["cv"].value_counts())

    num_cols = ["age", "past_purchases", "avg_spend"]
    cat_cols_s = ["gender_code", "coupon_encoded"]
    cat_cols_t = ["gender_code"]
    target_col = "cv"

    df_train_val, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=random_seed,
        stratify=df[target_col],
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.2,
        random_state=random_seed,
        stratify=df_train_val[target_col],
    )

    # S-learner
    s_learner = lgb.LGBMClassifier(
        objective="binary",
        num_iterations=1000,
        learning_rate=0.01,
        num_leaves=31,
        random_state=random_seed,
    )

    s_learner.fit(
        X=df_train[num_cols + cat_cols_s],
        y=df_train[target_col],
        eval_set=[(df_val[num_cols + cat_cols_s], df_val[target_col])],
        eval_metric=["average_precision", "auc", "binary_logloss"],
        categorical_feature=cat_cols_s,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=1),
        ],
    )

    coupon_map = df_train.groupby("coupon")["coupon_encoded"].first().to_dict()
    df_test_score = df_test.copy()
    for coupon_name, _ in coupon_settings.items():
        df_test_score["coupon_encoded"] = coupon_map[coupon_name]
        y_proba_test = s_learner.predict_proba(df_test_score[num_cols + cat_cols_s])[
            :, 1
        ]
        df_test_score[f"s_cvr_{coupon_name}"] = y_proba_test

    df_test = assign_coupon_by_score(df_test_score, df_test, "s_cvr", "s_score")

    # T-learner
    t_learner_models = {}
    for coupon in coupon_types:
        df_train_t = df_train.loc[df_train["coupon"] == coupon]
        df_val_t = df_val.loc[df_val["coupon"] == coupon]
        t_learner_model = lgb.LGBMClassifier(
            objective="binary",
            num_iterations=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=random_seed,
        )
        t_learner_model.fit(
            X=df_train_t[num_cols + cat_cols_t],
            y=df_train_t[target_col],
            eval_set=[(df_val_t[num_cols + cat_cols_t], df_val_t[target_col])],
            eval_metric=["average_precision", "auc", "binary_logloss"],
            categorical_feature=cat_cols_t,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1),
            ],
        )
        t_learner_models[coupon] = t_learner_model

    df_test_score = df_test.copy()
    for coupon, model in t_learner_models.items():
        y_proba_test = model.predict_proba(df_test_score[num_cols + cat_cols_t])[:, 1]
        df_test_score[f"t_cvr_{coupon}"] = y_proba_test

    df_test = assign_coupon_by_score(df_test_score, df_test, "t_cvr", "t_score")

    df_uplift = pd.DataFrame({"population": np.arange(1, len(df_test) + 1)})
    df_uplift["population_pct"] = df_uplift["population"] / len(df_uplift) * 100
    df_uplift = calc_cumulative_uplift(
        df_test,
        df_uplift,
        "s_score",
        "s_cum_cvr_uplift",
        "s_cum_cv_uplift",
    )
    df_uplift = calc_cumulative_uplift(
        df_test,
        df_uplift,
        "t_score",
        "t_cum_cvr_uplift",
        "t_cum_cv_uplift",
    )
    random_uplift = (
        len(df_test) * df_test.loc[df_test["coupon"] == "B", "cv"].mean()
        - len(df_test) * df_test.loc[df_test["coupon"] == "A", "cv"].mean()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_uplift, x="population_pct", y="s_cum_cv_uplift", label="s_learner"
    )
    sns.lineplot(
        data=df_uplift, x="population_pct", y="t_cum_cv_uplift", label="t_learner"
    )
    plt.plot([0, 100], [0, random_uplift], "--", label="baseline")
    plt.xlabel("Population Percentile (%)")
    plt.ylabel("Cumulative CV Uplift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "uplift_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
