# Test Chronos-2 time series forecasting

# ==========================================================================
# Import libraries
# ==========================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline


# ==========================================================================
# Define functions
# ==========================================================================
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
    ts_context = context_df.query(f"{id_column} == @timeseries_id").set_index(
        timestamp_column
    )[target_column]
    ts_pred = pred_df.query(
        f"{id_column} == @timeseries_id and target_name == @target_column"
    ).set_index(timestamp_column)[["0.1", "predictions", "0.9"]]
    ts_ground_truth = test_df.query(f"{id_column} == @timeseries_id").set_index(
        timestamp_column
    )[target_column]

    last_date = ts_context.index.max()
    start_idx = max(0, len(ts_context) - history_length)
    plot_cutoff = ts_context.index[start_idx]
    ts_context = ts_context[ts_context.index >= plot_cutoff]
    ts_pred = ts_pred[ts_pred.index >= plot_cutoff]
    ts_ground_truth = ts_ground_truth[ts_ground_truth.index >= plot_cutoff]

    fig = plt.figure(figsize=(12, 3))
    ax = fig.gca()
    ts_context.plot(ax=ax, label=f"historical {target_column}", color="xkcd:azure")
    ts_ground_truth.plot(
        ax=ax, label=f"future {target_column} (ground truth)", color="xkcd:grass green"
    )
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
    ax.set_title(f"{target_column} forecast for {timeseries_id} ({title_suffix})")
    fig.show()
    fig.savefig(
        f"./figures/{target_column}_{timeseries_id}_{title_suffix.replace(' ', '-')}.png",
        bbox_inches="tight",
    )


def retail_demand_forecasting(pipeline: Chronos2Pipeline):

    print("=== Running retail demand forecasting ===")

    # Retail forecasting configuration
    target = "Sales"  # Column name containing sales values to forecast
    prediction_length = 13  # Number of days to forecast ahead
    id_column = "id"  # Column identifying different products/stores
    timestamp_column = "timestamp"  # Column containing datetime information
    timeseries_id = "1"  # Specific time series to visualize (product/store ID)

    # Load historical sales and past values of covariates
    sales_context_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet"
    )
    sales_context_df[timestamp_column] = pd.to_datetime(
        sales_context_df[timestamp_column]
    )
    print(
        "The number of unique time series (products/stores) in the sales context dataframe is:",
        sales_context_df[id_column].nunique(),
    )
    print("Sales context dataframe shape:", sales_context_df.shape)
    print(sales_context_df.head())

    # Load future values of covariates
    sales_test_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet"
    )
    sales_test_df[timestamp_column] = pd.to_datetime(sales_test_df[timestamp_column])
    print("Sales test dataframe shape:", sales_test_df.shape)
    print(sales_test_df.head())

    # future
    sales_future_df = sales_test_df.drop(columns=target)
    print("Sales future dataframe shape:", sales_future_df.shape)
    print(sales_future_df.head())

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
    print(sales_pred_df.head())

    # Visualize forecast with covariates
    plot_forecast(
        sales_context_df,
        sales_pred_df,
        sales_test_df,
        target_column=target,
        timeseries_id=timeseries_id,
        title_suffix="with covariates",
    )

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
        title_suffix="without covariates",
    )

    # Fine-tuing API
    # --------------
    # Prepare data for fine-tuning using the retail sales dataset
    known_covariates = ["Open", "Promo", "SchoolHoliday", "StateHoliday"]
    past_covariates = ["Customers"]

    train_inputs = []
    for item_id, group in sales_context_df.groupby("id"):
        train_inputs.append(
            {
                "target": group[target].values,
                "past_covariates": {
                    col: group[col].values for col in past_covariates + known_covariates
                },
                # Future values of covariates are not used during training.
                # However, we need to include their names to indicate that these columns will be available at prediction time
                "future_covariates": {col: None for col in known_covariates},
            }
        )

    # Fine-tune the model by default full fine-tuning will be performed
    # -----------------------------------------------------------------
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=13,
        num_steps=1000,
        learning_rate=1e-5,
        batch_size=32,
        logging_steps=100,
    )

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
        title_suffix="full fine-tuned",
    )

    # Fine-tune the model with LoRA
    # -----------------------------
    lora_finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=13,
        num_steps=1000,
        learning_rate=1e-4,
        batch_size=32,
        logging_steps=100,
        finetune_mode="lora",
    )

    # Use the LoRA fine-tuned model for predictions
    # ---------------------------------------------
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
        title_suffix="LoRA fine-tuned",
    )


def m4_dataset_forecasting(pipeline: Chronos2Pipeline) -> pd.DataFrame:

    print("=== Running M4 dataset forecasting ===")

    # Load data as a long-format pandas data frame
    # --------------------------------------------
    context_df = pd.read_csv(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv"
    )
    print("Input dataframe shape:", context_df.shape)
    print(context_df.head())

    # predict
    # -------
    pred_df = pipeline.predict_df(
        context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]
    )

    print("Output dataframe shape:", pred_df.shape)
    print(pred_df.head())

    return context_df


def energy_price_forecasting(pipeline: Chronos2Pipeline):

    print("=== Running energy price forecasting ===")

    # Energy price forecasting configuration
    # --------------------------------------
    target = "target"  # Column name containing the values to forecast (energy prices)
    prediction_length = 24  # Number of hours to forecast ahead
    id_column = "id"  # Column identifying different time series (countries/regions)
    timestamp_column = "timestamp"  # Column containing datetime information
    timeseries_id = "DE"  # Specific time series to visualize (Germany)

    # Load historical energy prices and past values of covariates
    # -----------------------------------------------------------
    energy_context_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
    )
    energy_context_df[timestamp_column] = pd.to_datetime(
        energy_context_df[timestamp_column]
    )
    print("Energy context dataframe shape:", energy_context_df.shape)
    print(energy_context_df.head())

    # Load future values of covariates
    # --------------------------------
    # test
    energy_test_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
    )
    energy_test_df[timestamp_column] = pd.to_datetime(energy_test_df[timestamp_column])
    print("Energy test dataframe shape:", energy_test_df.shape)
    print(energy_test_df.head())

    # future
    energy_future_df = energy_test_df.drop(columns=target)
    print("Energy future dataframe shape:", energy_future_df.shape)
    print(energy_future_df.head())

    # Generate predictions with covariates
    # ------------------------------------
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
    print(energy_pred_df.head())

    # Visualize forecast with covariates
    plot_forecast(
        energy_context_df,
        energy_pred_df,
        energy_test_df,
        target_column=target,
        timeseries_id=timeseries_id,
        title_suffix="with covariates",
    )

    # Compare with forecasting without covariates
    # --------------------------------------------
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
        title_suffix="without covariates",
    )


def cross_learning_covariates_api(pipeline: Chronos2Pipeline, context_df: pd.DataFrame):

    print("=== Running cross-learning covariates API test ===")

    # Example: Enable cross-learning for joint prediction
    # ---------------------------------------------------
    # This assigns the same group ID to all time series, allowing information sharing
    joint_pred_df = pipeline.predict_df(
        context_df,
        prediction_length=24,
        quantile_levels=[0.1, 0.5, 0.9],
        cross_learning=True,  # Enable cross-learning
        batch_size=100,
    )
    print("Joint prediction dataframe shape:", joint_pred_df.shape)
    print(joint_pred_df.head())

    # Advanced Numpy/torch API
    # ------------------------
    # Univariate forecasting
    inputs = np.random.randn(32, 1, 100)
    quantiles, mean = pipeline.predict_quantiles(
        inputs, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]
    )
    print("Univariate output shapes:", quantiles[0].shape, mean[0].shape)

    # Multivariate forecasting
    inputs = np.random.randn(32, 3, 512)
    quantiles, mean = pipeline.predict_quantiles(
        inputs, prediction_length=48, quantile_levels=[0.1, 0.5, 0.9]
    )
    print("Multivariate output shapes:", quantiles[0].shape, mean[0].shape)

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
    print(
        "Univariate with covariates output shapes:", quantiles[0].shape, mean[0].shape
    )

    # Multivariate forecasting with categorical covariates
    prediction_length = 96
    inputs = [
        {
            "target": np.random.randn(2, 1000),
            "past_covariates": {
                "temperature": np.random.randn(1000),
                "weather_type": np.random.choice(
                    ["sunny", "cloudy", "rainy"], size=1000
                ),
            },
            "future_covariates": {
                "temperature": np.random.randn(prediction_length),
                "weather_type": np.random.choice(
                    ["sunny", "cloudy", "rainy"], size=prediction_length
                ),
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


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":

    # Use only 1 GPU if available
    # ---------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load the Chronos-2 pipeline
    # ---------------------------
    #
    # GPU recommended for faster inference, but CPU
    # is also supported using device_map="cpu"
    #
    # Note: Cannot use Windows symlinks, use "s3://autogluon/chronos-2/" for
    # windows. "amazon/chronos-2" for linux WSL users should point python to
    # system certificates (see .zshrc)
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda"
    )

    # Univeriate forecasting on M4 dataset
    # -------------------------------------
    m4_dataset_forecasting(pipeline)

    # Forecasting with covariates on energy price dataset
    # ---------------------------------------------------
    energy_price_forecasting(pipeline)

    # Forecasting with covariates on retail demand dataset
    # ------------------------------------------------------
    retail_demand_forecasting(pipeline)

# [EOF]
