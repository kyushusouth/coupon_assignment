import random

import pandas as pd

from ml_module import MLModule


def assign_coupon_to_users(
    ml_module: MLModule,
    test_pred_df: pd.DataFrame,
    segments: list[str],
    segment_n_users_map: dict[str, int],
    assign_solution: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """クーポンをユーザーに割り当てる

    Args:
        ml_module: MLModuleのインスタンス
        test_pred_df: 機械学習モデルによる予測結果を格納したデータ
        segments: セグメントIDリスト
        segment_n_users_map: セグメントIDをキー、セグメントごとのユーザー数をバリューとする辞書
        assign_solution: 最適化で得られたセグメントごとのクーポン付与割合

    Returns:
        割り当て結果
    """
    assign_rates = []
    for s in segments:
        for c in ml_module.coupon_map.keys():
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

    return test_assign_df, coupon_assign_df


def calculate_metrics(
    ml_module: MLModule, test_assign_df: pd.DataFrame
) -> tuple[float, float, float]:
    """評価指標を計算する

    Args:
        ml_module: MLModuleのインスタンス
        test_assign_df: ユーザーにクーポンを割り当てた結果

    Returns:
        割り当て結果から推定されるCV数、コスト、CPA
    """
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
            * x["assigned_coupon_type"].map(ml_module.coupon_map),
        )
    )
    estimated_cv = coupon_result_df["estimated_cv"].sum()
    estimated_cost = coupon_result_df["estimated_cost"].sum()
    estimated_cpa = estimated_cost / estimated_cv
    return estimated_cv, estimated_cost, estimated_cpa
