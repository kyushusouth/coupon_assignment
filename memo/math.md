# 3 パターンのクーポンを割り当てる問題

## 概要

クーポン 1 は 1000 円、クーポン 2 は 2000 円、クーポン 3 は 3000 円の割引を提供する。
ここではこれまでクーポン 2 が全ユーザーに配布されていた状況を仮定する。
クーポンの割り当てを最適化することにより、CV 数の最大化を目指す。

## 最適化問題

### 表記

$U$: ユーザー集合
$S$: ユーザーセグメント集合
$A$: クーポン集合
$x_{s,a}$: セグメント$s$へのクーポン$a$の配布率
$r_{s,a}$: セグメント$s$にクーポン$a$を配布した時に CV する確率
$p_{s,a}$: セグメント$s$にクーポンを$a$を配布して CV した際にクーポンを利用する確率（クーポンきっかけで興味を持ったが、結果的にクーポンを使わないで CV するケースも考えられる）
$N_{s}$: セグメント$s$に属するユーザー数
$C_{a}$: クーポン$a$を使って CV された際に発生するコスト
$B$: 予算

### アイデア

```math
\begin{align*}
    \text{Maximize} \quad & \sum_{s \in S} (x_{s, a_{3}} N_{s} (r_{s, a_{3}} - r_{s, a_{2}}) - x_{s, a_{1}} N_{s} (r_{s, a_{2}} - r_{s, a_{1}})) \\
    \text{s.t.} \quad & x_{s, a} \in [0, 1] \\
    & \sum_{a \in A} x_{s, a} = 1,\ \forall s \in S \\
    & \sum_{s \in S} \sum_{a \in A} x_{s, a} N_{s} r_{s, a} p_{s, a} C_{a} \le B \\
    & x_{s, a} \ge 0.1,\ \forall s \in S, \forall a \in A
\end{align*}
```

決定変数は$x_{s,a}$である。

目的関数は、クーポン 2 からクーポン 3 にする（割引額を増やす）ことによって増加する CV 数の最大化と、クーポン 2 からクーポン 1 にする（割引額を減らす）ことによって減少する CV 数の最小化の両方を考慮したものである。

制約条件は以下の通りである。

- 任意のセグメントに属するユーザーに対してなんらかのクーポンを一つ割り当てる。
- 期待コストが予算以下になる。
- 任意のセグメントに対し、すべてのクーポンを最低 10%配布する。

### 簡略化

上記の最適化問題は、下記の最適化問題に変形できる。

```math
\begin{align*}
    \text{Maximize} \quad & \sum_{s \in S} \sum_{a \in A} x_{s, a}N_{s}r_{s, a} \\
    \text{s.t.} \quad & x_{s, a} \in [0, 1] \\
    & \sum_{a \in A} x_{s, a} = 1,\ \forall s \in S \\
    & \sum_{s \in S} \sum_{a \in A} x_{s, a} N_{s} r_{s, a} p_{s, a} C_{a} \le B \\
    & x_{s, a} \ge 0.1,\ \forall s \in S, \forall a \in A
\end{align*}
```

実際、任意のセグメント$s \in S$に対して

```math
\begin{align*}
& (x_{s, a_{3}} N_{s} (r_{s, a_{3}} - r_{s, a_{2}}) - x_{s, a_{1}} N_{s} (r_{s, a_{2}} - r_{s, a_{1}})) \\
& = N_{s} (x_{s, a_{3}}(r_{s, a_{3}} - r_{s, a_{2}}) - x_{s, a_{1}}(r_{s, a_{2}} - r_{s, a_{1}})) \\
& = N_{s} (x_{s, a_{3}}r_{s, a_{3}} - x_{s, a_{3}}r_{s, a_{2}} - x_{s, a_{1}}r_{s, a_{2}} + x_{s, a_{1}}r_{s, a_{1}}) \\
& = N_{s} (x_{s, a_{3}}r_{s, a_{3}} + x_{s, a_{1}}r_{s, a_{1}} - r_{s, a_{2}}(x_{s, a_{3}} + x_{s, a_{1}})) \\
& = N_{s} (x_{s, a_{3}}r_{s, a_{3}} + x_{s, a_{1}}r_{s, a_{1}} - r_{s, a_{2}}(1 - x_{s, a_{2}})) \\
& = N_{s} \left(\sum_{a \in A} x_{s, a}r_{s, a} - r_{s, a_{2}}\right) \\
& = \sum_{a \in A} x_{s, a}N_{s}r_{s, a} - N_{s}r_{s, a_{2}} \\
\end{align*}
```

となる。5 行目の変形には制約条件の$\sum_{a \in A} x_{s, a} = 1$を用いた。目的関数から定数項$-N_{s}r_{s, a_{2}}$を除いても最適解は変わらないため、変形に問題がないと言える。

## ユーザーへのクーポン割り当て

最適化問題をソルバーを用いて解くことで、すべてのセグメント$s \in S$に対するクーポン$a \in A$の最適な配布割合$x_{s, a}$が得られる。ユーザーへのクーポンの割り当ては、配布割合$x_{u, s}$に基づいて確率的に行う。確率的な割り当てを行うことで、後に得られるログデータを用いた機械学習モデルの学習をバイアスなく行うことができる。
機械学習モデルを学習する際、目的関数は

```math
\mathcal{L}_{\text{ideal}}(\theta) = \frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} l(y, f_{\theta}(u, a))
```

である。また、

```math
l(y, f_{\theta}(u, a)) = y\log f_{\theta}(u, a) + (1 - y)\log(1 - f_{\theta}(u, a))
```

である。$y \in \{0, 1\}$は CV したかどうかの実測値、$f_{\theta}(u, a) \in [0, 1]$は機械学習モデルを用いて計算された予測確率である。実際に観測できるのは、各ユーザーに対してある 1 つのクーポンを割り当てた際の$y$の値だけだから、$\mathcal{L}_{\text{ideal}}(\theta)$を実際に計算することは不可能である。ここで、ユーザー$u$にクーポン$a$を割り当てた時の$y$の値が観測された場合に$O_{u, a} = 1$、そうでないときに$O_{u, a} = 0$として$O_{u, a}$を定義する。$y$の値が観測されたユーザーとクーポンの集合を$\mathcal{O} = \{(u, a) | O_{u, a} = 1\}$とすると、経験誤差は

```math
\mathcal{L}_{\text{naive}}(\theta) = \frac{1}{|\mathcal{O}|} \sum_{(u, a) \in \mathcal{O}} l(y, f_{\theta}(u, a))
```

となる。これは$\mathcal{L}_{\text{ideal}}(\theta)$に対する不偏推定量ではない [1, 2]。一方、傾向スコア$P(a|u) = P(O_{u, a} = 1)$の逆数で重み付けを行った

```math
\mathcal{L}_{\text{ips}}(\theta) = \frac{1}{|U||A|} \sum_{(u, a) \in \mathcal{O}} \frac{l(y, f_{\theta}(u, a))}{P(a|u)}
```

は、$\mathcal{L}_{\text{ideal}}(\theta)$に対する不偏推定量である。実際、

```math
\begin{align*}
\mathbb{E}[\mathcal{L}_{\text{ips}}(\theta)] & = \mathbb{E}\left[\frac{1}{|U||A|} \sum_{(u, a) \in \mathcal{O}} \frac{l(y, f_{\theta}(u, a))}{P(a|u)}\right] \\
& = \mathbb{E}\left[\frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} \frac{O_{u, a} l(y, f_{\theta}(u, a))}{P(a|u)}\right] \\
& = \frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} \mathbb{E}\left[\frac{O_{u, a} l(y, f_{\theta}(u, a))}{P(a|u)}\right] \\
& = \frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} P(O_{u, a} = 1) \frac{l(y, f_{\theta}(u, a))}{P(a|u)} \\
& = \frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} P(a|u) \frac{l(y, f_{\theta}(u, a))}{P(a|u)} \\
& = \frac{1}{|U||A|} \sum_{u \in U}\sum_{a \in A} l(y, f_{\theta}(u, a)) \\
& = \mathcal{L}_{\text{ideal}}(\theta)
\end{align*}
```

となる。従って、ユーザー$u$に対するクーポン$a$の割り当てを配布割合$x_{u, s}$に基づいて確率的に行い、この確率$P(a|u)$をログに残しておけば、$\mathcal{L}_{\text{ips}}(\theta)$を損失関数に用いることで、機械学習モデルをバイアスなく学習させることができる。

## 参考文献

[1] [反実仮想機械学習を用いたタクシーの乗車数予測と配置最適化](https://orsj.org/wp-content/corsj/or66-2/or66_2_66.pdf)
[2] [Recommendations as Treatments: Debiasing Learning and Evaluation](https://proceedings.mlr.press/v48/schnabel16.pdf)
