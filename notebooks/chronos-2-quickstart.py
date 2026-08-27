# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Getting Started with Chronos-2
#
# [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
# https://colab.research.google.com/github/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
#
#
#
# **Chronos-2** is a foundation model for time series forecasting that builds on [Chronos](https://arxiv.org/abs/2403.07815) and [Chronos-Bolt](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/). It offers significant improvements in capabilities and can handle diverse forecasting scenarios not supported by earlier models.
#
# | Capability | Chronos | Chronos-Bolt | Chronos-2 |
# |------------|---------|--------------|-----------|
# | Univariate Forecasting | ✅ | ✅ | ✅ |
# | Cross-learning across items | ❌ | ❌ | ✅ |
# | Multivariate Forecasting | ❌ | ❌ | ✅ |
# | Past-only (real/categorical) covariates | ❌ | ❌ | ✅ |
# | Known future (real/categorical) covariates | 🧩 | 🧩 | ✅ |
# | Fine-tuning support | ✅ | ✅ | ✅ |
# | Max. Context Length | 512 | 2048 | 8192 |
#
# 🧩 Chronos/Chronos-Bolt do not natively support future covariates, but they can be combined with external covariate regressors (see [AutoGluon tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html#incorporating-the-covariates)). This only models per-timestep effects, not effects across time. In contrast, Chronos-2 supports all covariate types natively.
#
# More details about Chronos-2 are available in the [technical report](https://www.arxiv.org/abs/2510.15821).

# %%
# # %pip install 'chronos-forecasting[extras]>=2.2' 'matplotlib'

# %%
import os

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set HF token in the environment. Do not hardcode it in production for security reasons.
# # !export HF_TOKEN="your_hf_token_here"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the Chronos-2 pipeline GPU recommended for faster inference, but CPU
# is also supported using device_map="cpu"
#
# Note: Cannot use Windows symlinks, use "s3://autogluon/chronos-2/" for
# windows. "amazon/chronos-2" for linux WSL users should point python to
# system certificates (see .zshrc)
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# %% [markdown]
# ## Univariate Forecasting
#
# We start with a simple univariate forecasting example using the pandas API.

# %%
# Load data as a long-format pandas data frame
context_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")
print("Input dataframe shape:", context_df.shape)
display(context_df.head())

# %%
pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])

print("Output dataframe shape:", pred_df.shape)
display(pred_df.head())

# %% [markdown]
# **predict_df** supports the following arguments:
# - `df`: Long-format DataFrame with id, timestamp, and target column(s)
# - `future_df`: Optional DataFrame with future covariates (columns present in both df and future_df are treated as known future covariates)
# - `id_column`: Column with time series identifiers (default: "item_id")
# - `timestamp_column`: Column with timestamps (default: "timestamp")
# - `target`: Target column name(s) to forecast (default: "target")
# - `prediction_length`: Number of steps to forecast
# - `quantile_levels`: Quantiles to compute (default: [0.1, 0.2, ..., 0.9])
#
# Returns a DataFrame with forecasts including point predictions and quantiles.

# %% [markdown]
# ## Forecasting with Covariates
#
# Chronos-2 can leverage covariates to improve forecast accuracy. We demonstrate this with two real-world examples.

# %% [markdown]
# ### Energy Price Forecasting
#
# Forecast hourly energy prices for the next day using historical prices and day-ahead forecasts of load (Ampirion Load Forecast) and renewable energy generation (PV+Wind Forecast).

# %%
# Energy price forecasting configuration
target = "target"  # Column name containing the values to forecast (energy prices)
prediction_length = 24  # Number of hours to forecast ahead
id_column = "id"  # Column identifying different time series (countries/regions)
timestamp_column = "timestamp"  # Column containing datetime information
timeseries_id = "DE"  # Specific time series to visualize (Germany)

# Load historical energy prices and past values of covariates
energy_context_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
)
energy_context_df[timestamp_column] = pd.to_datetime(energy_context_df[timestamp_column])
print("Energy context dataframe shape:", energy_context_df.shape)
display(energy_context_df.head())

# Load future values of covariates
# test
energy_test_df = pd.read_parquet(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
)
energy_test_df[timestamp_column] = pd.to_datetime(energy_test_df[timestamp_column])
print("Energy test dataframe shape:", energy_test_df.shape)
display(energy_test_df.head())

# future
energy_future_df = energy_test_df.drop(columns=target)
print("Energy future dataframe shape:", energy_future_df.shape)
display(energy_future_df.head())

# %%
# Generate predictions with covariates
energy_pred_df = pipeline.predict_df(
    energy_context_df,
    future_df=energy_future_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
print("Energy prediction dataframe shape:", energy_pred_df.shape)
display(energy_pred_df.head())

# %%
# Visualization helper function
def plot_forecast(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    timeseries_id: str,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    history_length: int = 256,
    title_suffix: str = "",
):
    ts_context = context_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target_column]
    ts_pred = pred_df.query(f"{id_column} == @timeseries_id and target_name == @target_column").set_index(
        timestamp_column
    )[["0.1", "predictions", "0.9"]]
    ts_ground_truth = test_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target_column]

    last_date = ts_context.index.max()
    start_idx = max(0, len(ts_context) - history_length)
    plot_cutoff = ts_context.index[start_idx]
    ts_context = ts_context[ts_context.index >= plot_cutoff]
    ts_pred = ts_pred[ts_pred.index >= plot_cutoff]
    ts_ground_truth = ts_ground_truth[ts_ground_truth.index >= plot_cutoff]

    fig = plt.figure(figsize=(12, 3))
    ax = fig.gca()
    ts_context.plot(ax=ax, label=f"historical {target_column}", color="xkcd:azure")
    ts_ground_truth.plot(ax=ax, label=f"future {target_column} (ground truth)", color="xkcd:grass green")
    ts_pred["predictions"].plot(ax=ax, label="forecast", color="xkcd:violet")
    ax.fill_between(
        ts_pred.index,
        ts_pred["0.1"],
        ts_pred["0.9"],
        alpha=0.7,
        label="prediction interval",
        color="xkcd:light lavender",
    )
    ax.axvline(x=last_date, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"{target_column} forecast for {timeseries_id} {title_suffix}")
    fig.show()

# %%
# Visualize forecast with covariates
plot_forecast(
    energy_context_df,
    energy_pred_df,
    energy_test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(with covariates)",
)

# %%
# Compare: forecast without covariates
energy_pred_no_cov_df = pipeline.predict_df(
    energy_context_df[[id_column, timestamp_column, target]],
    future_df=None,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)

plot_forecast(
    energy_context_df,
    energy_pred_no_cov_df,
    energy_test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(without covariates)",
)

# %% [markdown]
# The comparison shows that Chronos-2 makes reasonable but imprecise predictions in univariate mode. However, with covariates, Chronos-2 effectively uses the load and renewable generation forecasts, producing significantly more accurate predictions.

# %% [markdown]
# ### Retail Demand Forecasting
#
# Forecast next quarter's weekly store sales using historical sales, historical customer footfall (Customers), and known covariates indicating store operation (Open), promotion periods (Promo), and holidays (SchoolHoliday, StateHoliday).

# %%
# Retail forecasting configuration
target = "Sales"  # Column name containing sales values to forecast
prediction_length = 13  # Number of days to forecast ahead
id_column = "id"  # Column identifying different products/stores
timestamp_column = "timestamp"  # Column containing datetime information
timeseries_id = "1"  # Specific time series to visualize (product/store ID)

# Load historical sales and past values of covariates
sales_context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet")
sales_context_df[timestamp_column] = pd.to_datetime(sales_context_df[timestamp_column])
print(
    "The number of unique time series (products/stores) in the sales context dataframe is:",
    sales_context_df[id_column].nunique(),
)
print("Sales context dataframe shape:", sales_context_df.shape)
display(sales_context_df.head())

# Load future values of covariates
sales_test_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet")
sales_test_df[timestamp_column] = pd.to_datetime(sales_test_df[timestamp_column])
print("Sales test dataframe shape:", sales_test_df.shape)
display(sales_test_df.head())

# future
sales_future_df = sales_test_df.drop(columns=target)
print("Sales future dataframe shape:", sales_future_df.shape)
display(sales_future_df.head())

# %%
# Generate predictions with covariates
sales_pred_df = pipeline.predict_df(
    sales_context_df,
    future_df=sales_future_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
print("Sales prediction dataframe shape:", sales_pred_df.shape)
display(sales_pred_df.head())

# %%
# Visualize forecast with covariates
plot_forecast(
    sales_context_df,
    sales_pred_df,
    sales_test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(with covariates)",
)

# %%
# Compare: forecast without covariates
sales_pred_no_cov_df = pipeline.predict_df(
    sales_context_df[[id_column, timestamp_column, target]],
    future_df=None,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)

plot_forecast(
    sales_context_df,
    sales_pred_no_cov_df,
    sales_test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(without covariates)",
)

# %% [markdown]
# Chronos-2's univariate forecast is nearly flat with high uncertainty. In contrast, the forecast with covariates leverages promotion and holiday information to capture the true sales dynamics over the forecast horizon.

# %% [markdown]
# ## Cross-Learning with Joint Prediction
#
# Chronos-2 supports **cross-learning** through the `cross_learning=True` parameter, which enables the model to share information across all time series in a batch during prediction. This can be particularly beneficial when forecasting multiple related time series with short historical context.

# %%
# Example: Enable cross-learning for joint prediction
# This assigns the same group ID to all time series, allowing information sharing
joint_pred_df = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    cross_learning=True,  # Enable cross-learning
    batch_size=100,
)

# %% [markdown]
# <div class="alert alert-info alert-light" role="alert">
#
# ### Important Considerations for Cross-Learning
#
# When using `cross_learning=True`, keep these caveats in mind:
#
# - **Task-dependent results**: Cross-learning may not always improve forecasts and could worsen performance for some tasks. Evaluate this feature for your specific use case.
#
# - **Batch size dependency**: Results become dependent on batch size. Very large batch sizes may not provide benefits as they deviate from the maximum group size used during pretraining. For optimal results, consider using a batch size around 100 (as used in the paper).
#
# - **Input homogeneity**: This feature works best with homogeneous inputs (e.g., multiple univariate time series of the same type). Mixing different task types may lead to unexpected behavior.
#
# - **Short context benefit**: Cross-learning is most helpful when individual time series have limited historical context, as the model can leverage patterns from related series in the batch.
#
# </div>

# %% [markdown]
# ## (Advanced) Numpy/torch API
#
# For advanced use cases, Chronos-2 provides a lower-level numpy/torch API via the `predict_quantiles` method.
#
# The `predict_quantiles` method accepts:
# - `inputs`: Time series to forecast (see formats below)
# - `prediction_length`: Number of timesteps to forecast
# - `quantile_levels`: List of quantiles to compute
#
# Two input formats are supported:
# 1. **3D array**: `(batch_size, num_variates, history_length)` for forecasting without covariates
# 2. **List of dicts**: Each dict contains:
#    - `target`: 1D or 2D array of shape `(history_length,)` or `(num_variates, history_length)`
#    - `past_covariates` (optional): Dict mapping covariate names to 1D arrays of length `history_length`
#    - `future_covariates` (optional): Dict mapping covariate names to 1D arrays of length `prediction_length`

# %%
# Univariate forecasting
inputs = np.random.randn(32, 1, 100)
quantiles, mean = pipeline.predict_quantiles(inputs, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
print("Univariate output shapes:", quantiles[0].shape, mean[0].shape)

# %%
# Multivariate forecasting
inputs = np.random.randn(32, 3, 512)
quantiles, mean = pipeline.predict_quantiles(inputs, prediction_length=48, quantile_levels=[0.1, 0.5, 0.9])
print("Multivariate output shapes:", quantiles[0].shape, mean[0].shape)

# %%
# Univariate forecasting with covariates
prediction_length = 64
inputs = [
    {
        "target": np.random.randn(200),
        "past_covariates": {
            "temperature": np.random.randn(200),
            "precipitation": np.random.randn(200),
        },
        "future_covariates": {"temperature": np.random.randn(prediction_length)},
    }
    for _ in range(16)
]
quantiles, mean = pipeline.predict_quantiles(
    inputs, prediction_length=prediction_length, quantile_levels=[0.1, 0.5, 0.9]
)
print("Univariate with covariates output shapes:", quantiles[0].shape, mean[0].shape)

# %%
# Multivariate forecasting with categorical covariates
prediction_length = 96
inputs = [
    {
        "target": np.random.randn(2, 1000),
        "past_covariates": {
            "temperature": np.random.randn(1000),
            "weather_type": np.random.choice(["sunny", "cloudy", "rainy"], size=1000),
        },
        "future_covariates": {
            "temperature": np.random.randn(prediction_length),
            "weather_type": np.random.choice(["sunny", "cloudy", "rainy"], size=prediction_length),
        },
    }
    for _ in range(10)
]
quantiles, mean = pipeline.predict_quantiles(
    inputs, prediction_length=prediction_length, quantile_levels=[0.1, 0.5, 0.9]
)
print(
    "Multivariate with categorical covariates output shapes:",
    quantiles[0].shape,
    mean[0].shape,
)

# %% [markdown]
# ## Fine-Tuning
#
# Chronos-2 supports fine-tuning on your own data. You may either fine-tune all weights of the model (_full fine-tuning_) or a [low rank adapter (LoRA)](https://huggingface.co/docs/peft/en/package_reference/lora), which significantly reduces the number of trainable parameters.
#
# <div class="alert alert-warning alert-light" role="alert">
#
# **Note:** Fine-tuning functionality is intended for advanced users. The default fine-tuning hyperparameters may not always improve accuracy for your specific use case. We recommend experimenting with different hyperparameters. In case of limited data (too few and/or too short series), fine-tuning may not improve over zero-shot (and may even worsen accuracy sometimes).
#
# </div>

# %% [markdown]
# ### Fine-Tuning API
#
# The `fit` method accepts:
# - `inputs`: Time series for fine-tuning (same format as predict_quantiles)
# - `finetune_mode`: `"full"` or `"lora"`
# - `lora_config`: The [`LoraConfig`](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig), in case `finetune_mode="lora"`
# - `prediction_length`: Forecast horizon for fine-tuning
# - `validation_inputs`: Optional validation data (same format as inputs)
# - `learning_rate`: Optimizer learning rate (default: 1e-6, we recommend a higher learning rate such as 1e-5 for LoRA)
# - `num_steps`: Number of training steps (default: 1000)
# - `batch_size`: Batch size for training (default: 256)
#
# Returns a new pipeline with the fine-tuned model.
#
# Please read the docstring for details about specific arguments.

# %%
from chronos.chronos2 import preprocess

# Prepare data for fine-tuning using the retail sales dataset
train_inputs = preprocess.from_data_frame(
    sales_context_df,
    target_columns=[target],
    prediction_length=prediction_length,
    id_column=id_column,
    timestamp_column=timestamp_column,
    known_covariates_names=["Open", "Promo", "SchoolHoliday", "StateHoliday"],
    # Remaining columns are treated as past-only covariates
)

# %% [markdown]
# Optionally, you can fine-tune with a larger than memory dataset without loading it all to RAM. Here is an example how to do this with a Hugging Face `datasets` library. 
#
# You will need to save your data to disk in batches to avoid loading it all to memory.

# %%
# # %pip install datasets
# import datasets

# path = "my_lazy_dataset"
# datasets.Dataset.from_list(train_inputs).save_to_disk(path)
# ds = datasets.load_dataset(path).with_format("torch")
# # Pass `ds` instead of `train_inputs` when calling `fit()`

# %%
# Fine-tune the model by default full fine-tuning will be performed
finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=13,
    num_steps=1000,
    learning_rate=1e-5,
    batch_size=32,
    logging_steps=100,
)

# %%
# Use the fine-tuned model for predictions
finetuned_pred_df = finetuned_pipeline.predict_df(
    sales_context_df,
    future_df=sales_future_df,
    prediction_length=13,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="Sales",
)

plot_forecast(
    sales_context_df,
    finetuned_pred_df,
    sales_test_df,
    target_column="Sales",
    timeseries_id="1",
    title_suffix="(full fine-tuned)",
)

# %%
# Fine-tune the model with LoRA
lora_finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=13,
    num_steps=1000,
    learning_rate=1e-4,
    batch_size=32,
    logging_steps=100,
    finetune_mode="lora",
)

# %%
# Use the LoRA fine-tuned model for predictions
lora_finetuned_pred_df = lora_finetuned_pipeline.predict_df(
    sales_context_df,
    future_df=sales_future_df,
    prediction_length=13,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="Sales",
)

plot_forecast(
    sales_context_df,
    lora_finetuned_pred_df,
    sales_test_df,
    target_column="Sales",
    timeseries_id="1",
    title_suffix="(LoRA fine-tuned)",
)
