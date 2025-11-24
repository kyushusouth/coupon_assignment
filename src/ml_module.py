from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


class MLModule:
    """機械学習周りを担うモジュール"""

    def __init__(
        self,
        result_dir: Path,
        random_seed: int,
        n_users: int,
        coupon_map: dict[str, int],
        target_col_cv: str,
        target_col_point: str,
        num_cols: list[str],
        cat_cols: list[str],
        uplift_map: dict[str, float],
        rng: np.random.Generator,
    ):
        """
        Args:
            result_dir: 結果を保存するディレクトリのパス
            random_seed: 乱数シード
            n_users: 総ユーザー数
            coupon_map: クーポン名をキー、割引額をバリューとする辞書
            target_col_cv: cvrモデルの目的変数に使う列名
            target_col_point: purモデルの目的変数に使う列名
            num_cols: 量的変数の列名リスト
            cat_cols: 質的変数の列名リスト
            uplift_map: ダミーデータ生成時に用いるクーポンごとのCVされやすさへの寄与度合い
            rng: 乱数ジェネレータ
        """
        self.result_dir = result_dir
        self.random_seed = random_seed
        self.n_users = n_users
        self.coupon_map = coupon_map
        self.target_col_cv = target_col_cv
        self.target_col_point = target_col_point
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.uplift_map = uplift_map
        self.rng = rng

        self.df = self._generate_dummy_data()
        self.train_df, self.val_df, self.test_df = self._train_val_test_split()

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        self._preprocess_data()

    def _generate_dummy_data(self) -> pd.DataFrame:
        """ダミーデータを生成する"""
        df = pd.DataFrame(
            {
                "user_id": range(self.n_users),
                "age": self.rng.integers(20, 60, self.n_users),
                "history_count": self.rng.poisson(3, self.n_users),
                "recency": self.rng.integers(1, 365, self.n_users),
            }
        )
        df["coupon_type"] = self.rng.choice(
            list(self.coupon_map.keys()), size=self.n_users
        )

        base_logit = (
            -3.0 + 0.01 * df["age"] + 0.1 * df["history_count"] - 0.005 * df["recency"]
        )
        df["uplift_logit"] = df["coupon_type"].map(self.uplift_map)

        prob_cv = 1 / (1 + np.exp(-(base_logit + df["uplift_logit"])))
        df["is_cv"] = np.random.binomial(1, prob_cv)

        df["is_use"] = 0
        mask_cv = df["is_cv"] == 1
        df.loc[mask_cv, "is_use"] = self.rng.binomial(1, 0.6, mask_cv.sum().item())
        return df

    def _train_val_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """データを学習用、検証用、テスト用に分割する

        Returns:
            学習データ、検証データ、テストデータのtuple
        """
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
        """前処理"""
        self.train_df[self.cat_cols] = self.encoder.fit_transform(
            self.train_df[self.cat_cols]
        )
        self.val_df[self.cat_cols] = self.encoder.transform(self.val_df[self.cat_cols])

    def _train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        model_name: str,
    ) -> lgb.LGBMClassifier:
        """モデルの学習

        Args:
            train_df: 学習データ
            val_df: 検証データ
            target_col: 目的変数の列名
            model_name: モデルの名称

        Returns:
            学習済みモデル
        """
        model = lgb.LGBMClassifier(
            objective="binary",
            num_iterations=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=self.random_seed,
            importance_type="gain",
        )
        model.fit(
            X=train_df[self.num_cols + self.cat_cols],
            y=train_df[target_col],
            eval_set=[
                (
                    val_df[self.num_cols + self.cat_cols],
                    val_df[target_col],
                )
            ],
            eval_metric=["binary_logloss"],
            categorical_feature=self.cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1),
            ],
        )

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=300)
        lgb.plot_metric(model, "binary_logloss", ax=ax)
        plt.tight_layout()
        plt.savefig(self.result_dir.joinpath(f"{model_name}_lr_curve.png"))
        plt.close()

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=300)
        lgb.plot_importance(model, ax=ax)
        plt.tight_layout()
        plt.savefig(self.result_dir.joinpath(f"{model_name}_feature_importance.png"))
        plt.close()

        explainer = shap.TreeExplainer(model)
        shap_samples = val_df[self.num_cols + self.cat_cols].sample(1000)
        shap_values = explainer.shap_values(shap_samples)

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=300)
        shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(self.result_dir.joinpath(f"{model_name}_shap_bar.png"))
        plt.close()

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=300)
        shap.summary_plot(shap_values, shap_samples, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig(self.result_dir.joinpath(f"{model_name}_shap_dot.png"))
        plt.close()
        return model

    def train(self):
        """CVRモデルとPURモデルの学習"""
        self.model_cvr = self._train_model(
            self.train_df,
            self.val_df,
            self.target_col_cv,
            "cvr",
        )
        self.model_point = self._train_model(
            self.train_df.loc[self.train_df["is_cv"] == 1],
            self.val_df.loc[self.val_df["is_cv"] == 1],
            self.target_col_point,
            "pur",
        )

    def predict(self) -> pd.DataFrame:
        """CVRモデルとPURモデルによる予測

        Returns:
            予測結果を格納したデータ
        """
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
