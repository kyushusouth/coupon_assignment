import datetime
import math
import random
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

sns.set_style("whitegrid")

rng = np.random.default_rng(42)


class Trainer:
    def __init__(self):
        self.random_seed = 42
        self.n_users = 1000000
        self.coupon_map = {"A": 1000, "B": 1500}
        self.target_col_cv = "is_cv"
        self.target_col_point = "is_use"
        self.num_cols = ["age", "history_count", "recency"]
        self.cat_cols = ["coupon_type"]

        self.df = self._generate_dummy_data()
        self.train_df, self.val_df, self.test_df = self._train_val_test_split()

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        self._preprocess_data()

    def _generate_dummy_data(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "user_id": range(self.n_users),
                "age": rng.integers(20, 60, self.n_users),
                "history_count": rng.poisson(3, self.n_users),
                "recency": rng.integers(1, 365, self.n_users),
            }
        )
        df["coupon_type"] = rng.choice(list(self.coupon_map.keys()), size=self.n_users)

        base_logit = (
            -3.0 + 0.01 * df["age"] + 0.1 * df["history_count"] - 0.005 * df["recency"]
        )
        uplift_map = {"A": 0.1, "B": 0.9}
        df["uplift_logit"] = df["coupon_type"].map(uplift_map)

        prob_cv = 1 / (1 + np.exp(-(base_logit + df["uplift_logit"])))
        df["is_cv"] = np.random.binomial(1, prob_cv)

        df["is_use"] = 0
        mask_cv = df["is_cv"] == 1
        df.loc[mask_cv, "is_use"] = rng.binomial(1, 0.6, mask_cv.sum().item())
        return df

    def _train_val_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        mskf = MultilabelStratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_seed
        )

        train_val_indices, test_indices = next(
            mskf.split(self.df, self.df[[self.target_col_cv, self.target_col_point]])
        )
        train_val_df = self.df.iloc[train_val_indices]
        test_df = self.df.iloc[test_indices]

        train_indices, val_indices = next(
            mskf.split(
                train_val_df,
                train_val_df[[self.target_col_cv, self.target_col_point]],
            )
        )
        train_df = train_val_df.iloc[train_indices]
        val_df = train_val_df.iloc[val_indices]
        return train_df, val_df, test_df

    def _preprocess_data(self):
        self.train_df[self.cat_cols] = self.encoder.fit_transform(
            self.train_df[self.cat_cols]
        )
        self.val_df[self.cat_cols] = self.encoder.transform(self.val_df[self.cat_cols])

    def train(self):
        self.model_cvr = lgb.LGBMClassifier(
            objective="binary",
            num_iterations=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=self.random_seed,
        )
        self.model_cvr.fit(
            X=self.train_df[self.num_cols + self.cat_cols],
            y=self.train_df[self.target_col_cv],
            eval_set=[
                (
                    self.val_df[self.num_cols + self.cat_cols],
                    self.val_df[self.target_col_cv],
                )
            ],
            eval_metric=["binary_logloss"],
            categorical_feature=self.cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1),
            ],
        )

        train_point_df = self.train_df.loc[self.train_df["is_cv"] == 1]
        val_point_df = self.val_df.loc[self.val_df["is_cv"] == 1]
        self.model_point = lgb.LGBMClassifier(
            objective="binary",
            num_iterations=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=self.random_seed,
        )
        self.model_point.fit(
            X=train_point_df[self.num_cols + self.cat_cols],
            y=train_point_df[self.target_col_point],
            eval_set=[
                (
                    val_point_df[self.num_cols + self.cat_cols],
                    val_point_df[self.target_col_point],
                )
            ],
            eval_metric=["binary_logloss"],
            categorical_feature=self.cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1),
            ],
        )

    def predict(self) -> pd.DataFrame:
        test_pred_dfs = []
        for coupon_type in self.coupon_map.keys():
            X_test = self.test_df[self.num_cols + self.cat_cols].copy()
            X_test["coupon_type"] = coupon_type
            X_test[self.cat_cols] = self.encoder.transform(X_test[self.cat_cols])
            test_pred_df = self.test_df[
                ["user_id", "coupon_type", "is_cv", "is_use"]
            ].copy()
            test_pred_df["pred_coupon_type"] = coupon_type
            test_pred_df["pred_cvr"] = self.model_cvr.predict_proba(X_test)[:, 1]
            test_pred_df["pred_pur"] = self.model_point.predict_proba(X_test)[:, 1]
            test_pred_dfs.append(test_pred_df)
        test_pred_df = pd.concat(test_pred_dfs)
        test_pred_df = pd.pivot(
            test_pred_df,
            index=["user_id", "coupon_type", "is_cv", "is_use"],
            columns=["pred_coupon_type"],
            values=["pred_cvr", "pred_pur"],
        )
        test_pred_df.columns = [f"{col[0]}_{col[1]}" for col in test_pred_df.columns]
        test_pred_df = test_pred_df.reset_index()
        return test_pred_df


def solve_cost_minimization_problem(
    segment_df: pd.DataFrame, coupon_map: dict[str, int], allowed_cv: int
):
    prob = pulp.LpProblem("Cost_Minimization", pulp.LpMinimize)

    coupon_types = list(coupon_map.keys())
    segments = segment_df["segment_id"].unique().tolist()
    segment_n_users_map = {}
    segment_cvr_map = {}
    segment_pur_map = {}
    for s in segments:
        row = segment_df.loc[(segment_df["segment_id"] == s)]
        segment_n_users_map[s] = row["n_users"].to_numpy().item()
        segment_cvr_map[s] = {}
        segment_pur_map[s] = {}
        for c in coupon_types:
            segment_cvr_map[s][c] = row[f"pred_cvr_{c}"].to_numpy().item()
            segment_pur_map[s][c] = row[f"pred_pur_{c}"].to_numpy().item()

    # 決定変数: セグメントsに対するクーポンcの配布率
    x = pulp.LpVariable.dicts(
        "x", (segments, coupon_types), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )

    # 目的関数: 期待コスト
    prob += pulp.lpSum(
        [
            x[s][c]
            * segment_n_users_map[s]
            * segment_cvr_map[s][c]
            * segment_pur_map[s][c]
            * coupon_map[c]
            for s in segments
            for c in coupon_types
        ]
    )

    # 制約条件1: すべてのユーザーになんらかのクーポンを一つ割り当てる
    for s in segments:
        prob += pulp.lpSum([x[s][c] for c in coupon_types]) == 1

    # 制約条件2: 期待CV数が許容されるCV数以上である
    prob += (
        pulp.lpSum(
            [
                x[s][c] * segment_n_users_map[s] * segment_cvr_map[s][c]
                for s in segments
                for c in coupon_types
            ]
        )
        >= allowed_cv
    )

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    return x


def assign_coupon_to_users(
    trainer: Trainer,
    test_pred_df: pd.DataFrame,
    segments: list[str],
    segment_n_users_map: dict[str, int],
    assign_solution: dict,
) -> pd.DataFrame:
    assign_rates = []
    for s in segments:
        for c in trainer.coupon_map.keys():
            assign_rates.append(
                {
                    "segment_id": s,
                    "coupon_type": c,
                    "assign_rate": assign_solution[s][c].value(),
                }
            )
    coupon_assign_df = pd.DataFrame(assign_rates)
    coupon_assign_df = coupon_assign_df.pivot(
        index="segment_id", columns="coupon_type", values="assign_rate"
    ).reset_index()

    # ユーザーへのクーポン割り当て
    test_assign_df = test_pred_df[
        ["user_id", "segment_id", "coupon_type", "is_cv", "is_use"]
    ].copy()
    for s in segments:
        row = coupon_assign_df.loc[coupon_assign_df["segment_id"] == s]
        assign_rate_A = row["A"].to_numpy().item()
        assign_rate_B = row["B"].to_numpy().item()
        n_users = segment_n_users_map[s]
        n_users_A = int(n_users * assign_rate_A)
        n_users_B = int(n_users * assign_rate_B)
        n_users_C = n_users - n_users_A - n_users_B
        coupons = ["A"] * n_users_A + ["B"] * n_users_B + ["C"] * n_users_C
        coupons = random.sample(coupons, len(coupons))
        test_assign_df.loc[
            test_assign_df["segment_id"] == s, "assigned_coupon_type"
        ] = coupons

    return test_assign_df


def calculate_metrics(
    trainer: Trainer, test_assign_df: pd.DataFrame
) -> tuple[float, float, float]:
    estimated_rate_df = (
        test_assign_df.loc[
            test_assign_df["coupon_type"] == test_assign_df["assigned_coupon_type"]
        ]
        .groupby("assigned_coupon_type")
        .agg(
            n_users=("user_id", "nunique"),
            cv_count=("is_cv", "sum"),
            point_use_count=("is_use", "sum"),
        )
        .reset_index()
        .assign(
            cvr=lambda x: x["cv_count"] / x["n_users"],
            pur=lambda x: x["point_use_count"] / x["cv_count"],
        )
    )
    coupon_result_df = (
        test_assign_df.groupby("assigned_coupon_type")
        .agg(n_users=("user_id", "nunique"))
        .reset_index()
        .merge(
            estimated_rate_df[["assigned_coupon_type", "cvr", "pur"]],
            how="left",
            on="assigned_coupon_type",
        )
        .assign(
            estimated_cv=lambda x: x["n_users"] * x["cvr"],
            estimated_cost=lambda x: x["n_users"]
            * x["cvr"]
            * x["pur"]
            * x["assigned_coupon_type"].map(trainer.coupon_map),
        )
    )
    estimated_cv = coupon_result_df["estimated_cv"].sum()
    estimated_cost = coupon_result_df["estimated_cost"].sum()
    estimated_cpa = estimated_cost / estimated_cv
    return estimated_cv, estimated_cost, estimated_cpa


def main():
    root_dir = Path(__file__).parent.parent
    result_dir = root_dir.joinpath(
        "result",
        datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9))
        ).strftime("%Y%m%d_%H%M%S"),
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer()
    trainer.train()
    test_pred_df = trainer.predict()

    test_pred_df["cvr_rank_A"] = pd.qcut(
        test_pred_df["pred_cvr_A"], 10, duplicates="drop"
    )
    test_pred_df["cvr_rank_B"] = pd.qcut(
        test_pred_df["pred_cvr_B"], 10, duplicates="drop"
    )
    test_pred_df["segment_id"] = (
        test_pred_df["cvr_rank_A"].astype(str)
        + "_"
        + test_pred_df["cvr_rank_B"].astype(str)
    )

    segment_df = (
        test_pred_df.groupby("segment_id")
        .agg(
            n_users=("user_id", "count"),
            pred_cvr_A=("pred_cvr_A", "mean"),
            pred_cvr_B=("pred_cvr_B", "mean"),
            pred_pur_A=("pred_pur_A", "mean"),
            pred_pur_B=("pred_pur_B", "mean"),
        )
        .reset_index(drop=False)
    )

    segments = segment_df["segment_id"].unique().tolist()
    segment_n_users_map = segment_df.groupby("segment_id")["n_users"].max().to_dict()

    cvr_B = test_pred_df.loc[test_pred_df["coupon_type"] == "B", "is_cv"].mean()
    pur_B = test_pred_df.loc[
        (test_pred_df["coupon_type"] == "B") & (test_pred_df["is_cv"] == 1), "is_use"
    ].mean()
    estimated_cv_B = test_pred_df["user_id"].nunique() * cvr_B
    estimated_cost_B = estimated_cv_B * pur_B * trainer.coupon_map["B"]
    estimated_cpa_B = estimated_cost_B / estimated_cv_B

    results = []
    for loss_ratio in range(21):
        loss_ratio /= 100
        allowed_cv = (1 - loss_ratio) * estimated_cv_B
        assign_solution = solve_cost_minimization_problem(
            segment_df, trainer.coupon_map, allowed_cv
        )

        if not assign_solution:
            results.append(
                {
                    "loss_ratio": loss_ratio,
                    "allowed_cv": allowed_cv,
                    "estimated_cv": None,
                    "estimated_cost": None,
                    "estimated_cpa": None,
                    "estimated_cv_B": estimated_cv_B,
                    "estimated_cost_B": estimated_cost_B,
                    "estimated_cpa_B": estimated_cpa_B,
                }
            )
            continue

        test_assign_df = assign_coupon_to_users(
            trainer, test_pred_df, segments, segment_n_users_map, assign_solution
        )
        estimated_cv, estimated_cost, estimated_cpa = calculate_metrics(
            trainer, test_assign_df
        )
        results.append(
            {
                "loss_ratio": loss_ratio,
                "allowed_cv": allowed_cv,
                "estimated_cv": estimated_cv,
                "estimated_cost": estimated_cost,
                "estimated_cpa": estimated_cpa,
                "estimated_cv_B": estimated_cv_B,
                "estimated_cost_B": estimated_cost_B,
                "estimated_cpa_B": estimated_cpa_B,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(result_dir.joinpath("result.csv"), index=False)


if __name__ == "__main__":
    main()
