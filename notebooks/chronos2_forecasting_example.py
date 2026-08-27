# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: tsf-fm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chronos-2 Forecasting Example

# %% [markdown]
# From https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file

# %% [markdown]
# ## Import Libraries
#

# %%
# Set HF token in the environment. Do not hardcode it in production for security reasons.
# export HF_TOKEN="your_hf_token_here"

import matplotlib.pyplot as plt
import pandas as pd  # requires: pip install 'pandas[pyarrow]'

from chronos import Chronos2Pipeline

# %% [markdown]
# ## Load Data
#

# %%
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2", device_map="cuda"
)

# Load historical target values and past values of covariates
context_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
)

test_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
)

# (Optional) Load future values of covariates
future_df = test_df.drop(columns="target")

# %% [markdown]
# ## Generate predictions with covariates
#

# %%
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantile for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)

# %% [markdown]
# ## Visualization

# %%
ts_context = context_df.set_index("timestamp")["target"].tail(256)
ts_pred = pred_df.set_index("timestamp")
ts_ground_truth = test_df.set_index("timestamp")["target"]

ts_context.plot(label="historical data", color="xkcd:azure", figsize=(12, 3))
ts_ground_truth.plot(label="future data (ground truth)", color="xkcd:grass green")
ts_pred["predictions"].plot(label="forecast", color="xkcd:violet")
plt.fill_between(
    ts_pred.index,
    ts_pred["0.1"],
    ts_pred["0.9"],
    alpha=0.7,
    label="prediction interval",
    color="xkcd:light lavender",
)
plt.legend()
plt.show()
