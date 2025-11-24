from pathlib import Path

import pandas as pd

result_dir = Path(__file__).parent.parent.joinpath("result")
df = pd.read_csv(result_dir.joinpath("20251124_185336", "result.csv"))
df["cost_ratio_estimated"] = df["estimated_cost"] / df["estimated_cost_B"]
df = df[["cost_ratio_estimated", "estimated_cv", "estimated_cv_B"]]
print(df.to_markdown(index=False))
