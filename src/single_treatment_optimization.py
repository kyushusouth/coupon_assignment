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
        uplift_map = {"A": 0.5, "B": 0.9}
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
            num_iterations=100,
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
            num_iterations=100,
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
            test_pred_df = self.test_df[["user_id"]].copy()
            test_pred_df["pred_coupon_type"] = coupon_type
            test_pred_df["pred_cvr"] = self.model_cvr.predict_proba(X_test)[:, 1]
            test_pred_df["pred_pur"] = self.model_point.predict_proba(X_test)[:, 1]
            test_pred_dfs.append(test_pred_df)
        test_pred_df = pd.concat(test_pred_dfs)
        test_pred_df = pd.pivot(
            test_pred_df,
            index=["user_id"],
            columns=["pred_coupon_type"],
            values=["pred_cvr", "pred_pur"],
        )
        test_pred_df.columns = [f"{col[0]}_{col[1]}" for col in test_pred_df.columns]
        test_pred_df = test_pred_df.reset_index()
        return test_pred_df


def solve_optimization_problem(
    segment_df: pd.DataFrame, coupon_map: dict[str, int], target_cv_count: float
) -> dict:
    prob = pulp.LpProblem("Coupon_Allocation_Optimization", pulp.LpMinimize)

    segments = segment_df["segment_id"].unique().tolist()
    coupon_types = list(coupon_map.keys())

    x = pulp.LpVariable.dicts(
        "x", (segments, coupon_types), lowBound=0, upBound=1, cat=pulp.LpBinary
    )

    # あるセグメントにあるクーポンを割り当てた時の期待コストと期待CV数
    costs = {}
    cvs = {}
    for s in segments:
        row = segment_df.loc[segment_df["segment_id"] == s]
        n = row["n_users"].to_numpy().item()
        costs[s] = {}
        cvs[s] = {}

        for c in coupon_types:
            cvr = row[f"pred_cvr_{c}"].to_numpy().item()
            pur = row[f"pred_pur_{c}"].to_numpy().item()
            unit_cost = coupon_map[c]
            costs[s][c] = n * cvr * pur * unit_cost
            cvs[s][c] = n * cvr

    # 目的関数: 総コストの最小化
    prob += pulp.lpSum([x[s][c] * costs[s][c] for s in segments for c in coupon_types])

    # 制約条件1: 各セグメントに対して、1つのクーポンを割り当てる
    for s in segments:
        prob += pulp.lpSum([x[s][c] for c in coupon_types]) == 1

    # 制約条件2: 総予約数が目標値以上である
    prob += (
        pulp.lpSum([x[s][c] * cvs[s][c] for s in segments for c in coupon_types])
        >= target_cv_count
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    total_cost = pulp.value(prob.objective)
    total_cv = sum(x[s][c].value() * cvs[s][c] for s in segments for c in coupon_types)
    return {
        "assignment": x,
        "cost": total_cost,
        "cv": total_cv,
        "status": pulp.LpStatus[prob.status],
    }


def main():
    trainer = Trainer()
    trainer.train()
    test_pred_df = trainer.predict()

    test_pred_df["uplift_score"] = (
        test_pred_df["pred_cvr_B"] - test_pred_df["pred_cvr_A"]
    )

    test_pred_df["base_cvr_rank"] = pd.qcut(
        test_pred_df["pred_cvr_A"], 10, duplicates="drop"
    )
    test_pred_df["uplift_rank"] = pd.qcut(
        test_pred_df["uplift_score"], 10, duplicates="drop"
    )
    test_pred_df["segment_id"] = (
        test_pred_df["base_cvr_rank"].astype("str")
        + "_"
        + test_pred_df["uplift_rank"].astype(str)
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

    total_cv_all_A = (segment_df["n_users"] * segment_df["pred_cvr_A"]).sum()
    total_cv_all_B = (segment_df["n_users"] * segment_df["pred_cvr_B"]).sum()
    max_possible_cv = (test_pred_df[["pred_cvr_A", "pred_cvr_B"]].max(axis=1)).sum()

    allowed_loss_ratios = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    results = []
    segments = segment_df["segment_id"].unique().tolist()
    coupon_types = list(trainer.coupon_map.keys())
    for ratio in allowed_loss_ratios:
        target_cv = max_possible_cv * (1 - ratio)
        res = solve_optimization_problem(segment_df, trainer.coupon_map, target_cv)

        assignment = res["assignment"]
        assignments = []
        for s in segments:
            for c in coupon_types:
                assignments.append(
                    {
                        "segment_id": s,
                        "coupon_type": c,
                        "is_assigned": assignment[s][c].value(),
                    }
                )
        point_assignment_df = pd.DataFrame(assignments)

        # テストデータにポイントを割り当てて、実測データからコストと予約数を計算する
        # ランダム割り当てとコスト、予約数、CPA=コスト/予約数を比較する

        if res:
            results.append(
                {
                    "allowed_loss_ratio": ratio,
                    "target_cv": target_cv,
                    "actual_cv": res["cv"],
                    "min_cost": res["cost"],
                    "cost_per_cv": res["cost"] / res["cv"] if res["cv"] > 0 else 0,
                }
            )

    result_df = pd.DataFrame(results)


if __name__ == "__main__":
    main()
