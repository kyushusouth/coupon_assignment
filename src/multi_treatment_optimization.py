import datetime
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns

from ml_module import MLModule
from utils import assign_coupon_to_users, calculate_metrics

sns.set_style("whitegrid")


def solve_cost_minimization_problem(
    segment_df: pd.DataFrame, coupon_map: dict[str, int], min_assign_rate: float
):
    """コストを最小化する

    Args:
        segment_df: ユーザーをセグメント分けしたデータ
        coupon_map: クーポン名をキー、割引額をバリューとする辞書
        min_assign_rate: 最低配布率

    Returns:
        最適化問題を解くことができた場合: セグメントに対するクーポンの付与割合
        最適化問題を解くことができなかった場合: None
    """
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

    # 制約条件2: 各セグメントにすべてのクーポンをmin_assign_rate以上割り当てる
    for s in segments:
        for c in coupon_types:
            prob += x[s][c] >= min_assign_rate

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    min_cost = math.ceil(pulp.value(prob.objective))
    return min_cost


def solve_cv_maximization_problem(
    segment_df: pd.DataFrame,
    coupon_map: dict[str, int],
    budget: int,
    min_assign_rate: float,
):
    """CV数を最大化する

    Args:
        segment_df: ユーザーをセグメント分けしたデータ
        coupon_map: クーポン名をキー、割引額をバリューとする辞書
        budget: 予算
        min_assign_rate: 最低配布率

    Returns:
        最適化問題を解くことができた場合: セグメントに対するクーポンの付与割合
        最適化問題を解くことができなかった場合: None
    """
    prob = pulp.LpProblem("CV_Maximization", pulp.LpMaximize)

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

    # 目的関数: 期待予約数
    prob += pulp.lpSum(
        [
            x[s][c] * segment_n_users_map[s] * segment_cvr_map[s][c]
            for s in segments
            for c in coupon_types
        ]
    )

    # 制約条件1: すべてのユーザーになんらかのクーポンを一つ割り当てる
    for s in segments:
        prob += pulp.lpSum([x[s][c] for c in coupon_types]) == 1

    # 制約条件2: 期待コストが予算を超えない
    prob += (
        pulp.lpSum(
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
        <= budget
    )

    # 制約条件3: 各セグメントにすべてのクーポンをmin_assign_rate以上割り当てる
    for s in segments:
        for c in coupon_types:
            prob += x[s][c] >= min_assign_rate

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    return x


def main():
    root_dir = Path(__file__).parent.parent
    result_dir = root_dir.joinpath(
        "result",
        datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9))
        ).strftime("%Y%m%d_%H%M%S"),
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    random_seed = 42
    n_users = 1000000
    coupon_map = {"A": 1000, "B": 1500, "C": 2000}
    target_col_cv = "is_cv"
    target_col_point = "is_use"
    num_cols = ["age", "history_count", "recency"]
    cat_cols = ["coupon_type"]
    uplift_map = {"A": 0.1, "B": 0.9, "C": 1.7}
    rng = np.random.default_rng(42)
    min_assign_rate = 0.1
    num_segment = 6

    ml_module = MLModule(
        result_dir,
        random_seed,
        n_users,
        coupon_map,
        target_col_cv,
        target_col_point,
        num_cols,
        cat_cols,
        uplift_map,
        rng,
    )
    ml_module.train()
    test_pred_df = ml_module.predict()

    test_pred_df["cvr_rank_A"] = pd.qcut(
        test_pred_df["pred_cvr_A"], num_segment, duplicates="drop"
    )
    test_pred_df["cvr_rank_B"] = test_pred_df.groupby("cvr_rank_A", observed=False)[
        "pred_cvr_B"
    ].transform(lambda x: pd.qcut(x, num_segment, duplicates="drop"))
    test_pred_df["cvr_rank_C"] = test_pred_df.groupby(
        ["cvr_rank_A", "cvr_rank_B"], observed=False
    )["pred_cvr_C"].transform(lambda x: pd.qcut(x, num_segment, duplicates="drop"))
    test_pred_df["segment_id"] = (
        test_pred_df["cvr_rank_A"].astype(str)
        + "_"
        + test_pred_df["cvr_rank_B"].astype(str)
        + "_"
        + test_pred_df["cvr_rank_C"].astype(str)
    )

    segment_df = (
        test_pred_df.groupby("segment_id")
        .agg(
            n_users=("user_id", "count"),
            pred_cvr_A=("pred_cvr_A", "mean"),
            pred_cvr_B=("pred_cvr_B", "mean"),
            pred_cvr_C=("pred_cvr_C", "mean"),
            pred_pur_A=("pred_pur_A", "mean"),
            pred_pur_B=("pred_pur_B", "mean"),
            pred_pur_C=("pred_pur_C", "mean"),
        )
        .reset_index(drop=False)
    )

    segment_actual_cv_df = (
        test_pred_df.groupby(["segment_id", "coupon_type"])
        .agg(cvr=("is_cv", "mean"))
        .reset_index()
        .pivot(index="segment_id", columns="coupon_type", values="cvr")
    )
    segment_actual_cv_df.columns = [
        f"actual_cvr_{col[0]}" for col in segment_actual_cv_df.columns
    ]
    segment_actual_cv_df = segment_actual_cv_df.reset_index()

    segment_actual_point_df = (
        test_pred_df.loc[test_pred_df["is_cv"] == 1]
        .groupby(["segment_id", "coupon_type"])
        .agg(pur=("is_use", "mean"))
        .reset_index()
        .pivot(index="segment_id", columns="coupon_type", values="pur")
    )
    segment_actual_point_df.columns = [
        f"actual_pur_{col[0]}" for col in segment_actual_point_df.columns
    ]
    segment_actual_point_df = segment_actual_point_df.reset_index()

    segment_mst_df = segment_df.merge(
        segment_actual_cv_df, how="left", on="segment_id"
    ).merge(segment_actual_point_df, how="left", on="segment_id")
    segment_mst_df.to_csv(result_dir.joinpath("segment_mst.csv"), index=False)

    for metric in ["cvr_A", "cvr_B", "cvr_C", "pur_A", "pur_B", "pur_C"]:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), dpi=300)
        sns.scatterplot(segment_mst_df, x=f"pred_{metric}", y=f"actual_{metric}", ax=ax)
        plt.tight_layout()
        plt.savefig(result_dir.joinpath(f"{metric}_scatter.png"))
        plt.close()

    # PUR予測モデルの学習が困難だったため、ログデータから集計した値を利用
    segment_df = segment_mst_df[
        [
            "segment_id",
            "n_users",
            "pred_cvr_A",
            "pred_cvr_B",
            "pred_cvr_C",
            "actual_pur_A",
            "actual_pur_B",
            "actual_pur_C",
        ]
    ]
    segment_df = segment_df.rename(
        columns={
            "actual_pur_A": "pred_pur_A",
            "actual_pur_B": "pred_pur_B",
            "actual_pur_C": "pred_pur_C",
        }
    )

    segments = segment_df["segment_id"].unique().tolist()
    segment_n_users_map = segment_df.groupby("segment_id")["n_users"].max().to_dict()

    cvr_B = test_pred_df.loc[test_pred_df["coupon_type"] == "B", "is_cv"].mean()
    pur_B = test_pred_df.loc[
        (test_pred_df["coupon_type"] == "B") & (test_pred_df["is_cv"] == 1), "is_use"
    ].mean()
    estimated_cv_B = test_pred_df["user_id"].nunique() * cvr_B
    estimated_cost_B = estimated_cv_B * pur_B * ml_module.coupon_map["B"]
    estimated_cpa_B = estimated_cost_B / estimated_cv_B

    min_cost = solve_cost_minimization_problem(
        segment_df, ml_module.coupon_map, min_assign_rate
    )

    coupon_assign_dfs = []
    results = []
    for budget in sorted(
        list(range(min_cost, 50000000, 2000000))
        + [
            estimated_cost_B,
            estimated_cost_B * 0.95,
            estimated_cost_B * 0.97,
            estimated_cost_B * 0.99,
        ]
    ):
        assign_solution = solve_cv_maximization_problem(
            segment_df, ml_module.coupon_map, budget, min_assign_rate
        )

        if not assign_solution:
            results.append(
                {
                    "budget": budget,
                    "estimated_cv": None,
                    "estimated_cost": None,
                    "estimated_cpa": None,
                    "estimated_cv_B": estimated_cv_B,
                    "estimated_cost_B": estimated_cost_B,
                    "estimated_cpa_B": estimated_cpa_B,
                }
            )
            continue

        test_assign_df, coupon_assign_df = assign_coupon_to_users(
            ml_module, test_pred_df, segments, segment_n_users_map, assign_solution
        )
        estimated_cv, estimated_cost, estimated_cpa = calculate_metrics(
            ml_module, test_assign_df
        )

        coupon_assign_df["budget"] = budget
        coupon_assign_dfs.append(coupon_assign_df)
        results.append(
            {
                "budget": budget,
                "estimated_cv": estimated_cv,
                "estimated_cost": estimated_cost,
                "estimated_cpa": estimated_cpa,
                "estimated_cv_B": estimated_cv_B,
                "estimated_cost_B": estimated_cost_B,
                "estimated_cpa_B": estimated_cpa_B,
            }
        )

    coupon_assign_df = pd.concat(coupon_assign_dfs)
    coupon_assign_df.to_csv(result_dir.joinpath("coupon_assign.csv"), index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(result_dir.joinpath("result.csv"), index=False)

    result_df["cost_ratio_estimated"] = (
        result_df["estimated_cost"] / result_df["estimated_cost_B"]
    )
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)
    plt.plot(
        result_df["cost_ratio_estimated"].values,
        result_df["estimated_cv"].values,
        marker="o",
        label="Optimized",
    )
    plt.plot([1], [estimated_cv_B], marker="x", label="All B")
    plt.xlabel("Cost Increase Rate")
    plt.ylabel("Estimated CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("cost_cv_curve.png"))


if __name__ == "__main__":
    main()
