import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns
import shap
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

random_seed = 42
sns.set_style("whitegrid")

budget = 100000000
n_users = 1000000
coupon_settings = {"A": 100, "B": 300, "C": 500, "D": 1000}
coupon_types = list(coupon_settings.keys())
n_clusters = 10000


def create_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    features = {
        "age": rng.integers(18, 71, n_users),
        "gender_code": rng.choice([0, 1], n_users, replace=True, p=[0.5, 0.5]),
        "past_purchases": rng.poisson(3, n_users),
        "avg_spend": np.clip(rng.normal(5000, 1500, n_users), 0, None).astype(int),
    }

    df = pd.DataFrame(features)
    df["coupon"] = rng.choice(coupon_types, n_users)
    df["discount_amount"] = df["coupon"].map(coupon_settings)

    beta_intercept = 0
    beta_age = -0.02  # 年齢が低い方がcvしやすい
    beta_gender = 0.1  # gender_code=1の方がcvしやすい
    beta_past_purchases = 0.2  # 過去購入回数が多い方がcvしやすい
    beta_avg_spend = 0.0001  # 過去平均支出額が高い方がcvしやすい
    tau_discount = 0.001  # 割引額が高い方がcvしやすい
    noise = rng.normal(0, 0.3, n_users)

    logit = (
        beta_intercept
        + df["age"] * beta_age
        + (df["gender_code"] == 1) * beta_gender
        + df["past_purchases"] * beta_past_purchases
        + df["avg_spend"] * beta_avg_spend
        + df["discount_amount"] * tau_discount
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

    num_cols = ["age", "past_purchases", "avg_spend"]
    cat_cols = ["gender_code", "coupon_encoded"]
    target_col = "cv"

    # 機械学習モデルによりcv確率を取得

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

    model = lgb.LGBMClassifier(
        objective="binary",
        num_iterations=1000,
        learning_rate=0.01,
        num_leaves=31,
        seed=random_seed,
    )

    model.fit(
        X=df_train[num_cols + cat_cols],
        y=df_train[target_col],
        eval_set=[(df_val[num_cols + cat_cols], df_val[target_col])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=1),
        ],
    )

    _ = lgb.plot_metric(model, metric="auc", figsize=(8, 6), dpi=300)
    plt.tight_layout()
    plt.savefig(result_dir / "learning_curve.png")
    plt.close()

    _ = lgb.plot_importance(model, importance_type="gain", figsize=(8, 6), dpi=300)
    plt.tight_layout()
    plt.savefig(result_dir / "feature_importance.png")
    plt.close()

    explainer = shap.TreeExplainer(model)
    shap_samples = df_test.sample(min(len(df_test), 1000))
    shap_values_test = explainer.shap_values(shap_samples[num_cols + cat_cols])
    plt.figure(figsize=(6, 10))
    shap.summary_plot(
        shap_values_test, shap_samples[num_cols + cat_cols], plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(result_dir / "shap_summary_plot_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 10))
    shap.summary_plot(
        shap_values_test, shap_samples[num_cols + cat_cols], plot_type="dot", show=False
    )
    plt.tight_layout()
    plt.savefig(result_dir / "shap_summary_plot_dot.png", bbox_inches="tight")
    plt.close()

    coupon_cat_map = (
        df[["coupon", "coupon_encoded"]]
        .groupby("coupon")["coupon_encoded"]
        .first()
        .to_dict()
    )
    df_test_base = df_test.drop(columns="coupon_encoded").reset_index(drop=True)
    df_test_base["user_index"] = df_test_base.index.values

    df_res_list = []
    for coupon_type in coupon_types:
        df_test_coupon = df_test_base.copy()
        df_test_coupon["coupon_encoded"] = coupon_cat_map[coupon_type]
        pred_proba = model.predict_proba(df_test_coupon[num_cols + cat_cols])[:, 1]
        df_res_tmp = df_test_base[["user_index"]].copy()
        df_res_tmp["coupon"] = coupon_type
        df_res_tmp["proba"] = pred_proba
        df_res_tmp["cost"] = pred_proba * coupon_settings[coupon_type]
        df_res_list.append(df_res_tmp)
    df_res = pd.concat(df_res_list)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        batch_size=10000,
        compute_labels=True,
        random_state=random_seed,
        verbose=1,
        max_iter=10000,
    )
    scaler = StandardScaler()
    user_features = scaler.fit_transform(df_test_base[num_cols])
    df_test_base["user_cluster"] = kmeans.fit_predict(user_features)

    df_res = df_res.merge(
        df_test_base[["user_index", "user_cluster"]].drop_duplicates(),
        on="user_index",
        how="inner",
    )

    cluster_n_user_map = (
        df_res.groupby("user_cluster")["user_index"].nunique().to_dict()
    )
    cluster_coupon_proba_map = (
        df_res.groupby(["user_cluster", "coupon"])["proba"].mean().to_dict()
    )
    cluster_coupon_cost_map = (
        df_res.groupby(["user_cluster", "coupon"])["cost"].mean().to_dict()
    )
    cluster_indices = df_res["user_cluster"].unique().tolist()

    problem = pulp.LpProblem("Coupon_Optimization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts(
        "assign", (cluster_indices, coupon_types), cat=pulp.LpBinary
    )

    problem += pulp.lpSum(
        cluster_n_user_map[cluster_index]
        * x[cluster_index][coupon]
        * cluster_coupon_proba_map[(cluster_index, coupon)]
        for cluster_index in cluster_indices
        for coupon in coupon_types
    )

    for cluster_index in cluster_indices:
        problem += pulp.lpSum(x[cluster_index][coupon] for coupon in coupon_types) == 1

    expected_cost = pulp.lpSum(
        cluster_n_user_map[cluster_index]
        * x[cluster_index][coupon]
        * cluster_coupon_cost_map[(cluster_index, coupon)]
        for cluster_index in cluster_indices
        for coupon in coupon_types
    )
    problem += expected_cost <= budget

    status = problem.solve()
    if status == pulp.LpStatusOptimal:
        actual_total_cv = df_test_base["cv"].sum()
        expected_total_cv = pulp.value(problem.objective)
        diff = expected_total_cv - actual_total_cv
        print(f"actual: {actual_total_cv}")
        print(f"expected: {expected_total_cv}")
        print(f"diff: {diff}")
        print(f"expected_cost: {pulp.value(expected_cost)}")


if __name__ == "__main__":
    main()
