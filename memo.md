# やること

## 表記の定義

$U$: ユーザー集合
$u \in U$: ユーザー
$x_{u}$: ユーザー u の特徴量
$A$: アクション集合
$a \in A$: アクション
$c_{a}$: アクションのコスト
$l_{u, a}$: ユーザー u にアクション a を割り当てるかを表す変数
$\hat{\pi}(a|x_{u})$: ユーザー u がアクション a を割り当てられた時に cv する確率
$p(\text{use}|a, x_{u})$: ユーザー u がアクション a を割り当てられ、cv した際に アクション a を実行する確率
$B$: 予算

## 定式化

```math
\begin{align*}
\max \quad & \sum_{u \in U} \sum_{a \in A} l_{u, a} \hat{\pi}(a|x_{u}) \\
\text{s.t.} \quad & \sum_{u \in U} \sum_{a \in A} l_{u, a} \hat{\pi}(a|x_{u}) p(\text{use}|a, x_{u}) c_{a} \le B \\
& \sum_{a \in A} l_{u, a} \le 1,\ \forall u \in U \\
& l_{u, a} \in \{0, 1\},\ \forall u \in U, \forall a \in A
\end{align*}
```
