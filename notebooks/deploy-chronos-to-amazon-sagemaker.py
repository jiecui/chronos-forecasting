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
#     display_name: ag
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deploy Chronos-2 to AWS with Amazon SageMaker

# %% [markdown]
# <div class="alert alert-info">
# <b>💡 Looking for a higher-level API?</b><br>
# This tutorial uses the SageMaker Python SDK directly, which means you'll write your own helpers to convert pandas DataFrames to and from the JSON request/response format. If you'd prefer a high-level API where you pass DataFrames directly and get forecasts back, check out <a href="https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html"><b>AutoGluon-Cloud</b></a> — it also covers serverless and batch inference with minimal setup.
# </div>

# %% [markdown]
# This notebook shows how to deploy **Chronos-2** to AWS using **Amazon SageMaker**.
#
# ### Why Deploy to SageMaker?
# Running models locally works for experimentation, but production use cases need reliability, scale, and integration into existing workflows. For example, you may need to generate forecasts for thousands of time series on a regular schedule, or integrate forecasts into applications that serve many users. SageMaker lets you deploy Chronos-2 to the cloud and access it from anywhere.
#
# ### Deployment Options
# This notebook covers three deployment modes on SageMaker:
#
# 1. **[Real-time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)**
#     - ✅ Highest throughput, consistently low latency, supports both GPU and CPU instances
#     - ✅ Simple setup via JumpStart
#     - ❌ By default, you pay for the time the endpoint is running (can be configured to [scale to zero](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html))
#
# 2. **[Serverless Inference (CPU only)](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)**
#     - ✅ Pay only for active inference time, no infrastructure management
#     - ✅ Cost-efficient for intermittent or unpredictable traffic
#     - ❌ Cold start latency on first request after idle, lowest throughput of all options
#     - ❌ More complex setup (requires repackaging model artifacts)
#
# 3. **[Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)**
#     - ✅ Pay only for active compute time, no persistent infrastructure
#     - ✅ Cost-efficient for large-scale batch prediction jobs
#     - ❌ Initialization takes severa minutes for each job (not for real-time use), requires data in S3
#     - ❌ More complex setup (requires repackaging model artifacts)
#
# **Reference benchmark** on a dataset with 1M rows (2000 time series with 500 observations each) and prediction length of 28:
# | Mode | Instance | Inference time (s) |
# |------|----------|------|
# | Real-time (GPU) | ml.g5.2xlarge | 18 |
# | Real-time (CPU) | ml.c5.4xlarge | 50 |
# | Serverless | 6GB memory | 120 |
# | Batch Transform | ml.c5.4xlarge | 60 (+200s setup) |
#
# We recommend starting with **Real-time Inference** as it offers the simplest setup and highest throughput. Consider Serverless or Batch Transform when you need to optimize costs and don't require GPU acceleration.
#
# For a complete specification of all supported request parameters, see the **Endpoint API Reference** at the end of this notebook.

# %% [markdown]
# <div class="alert alert-info">
# <b>ℹ️ New to Chronos-2?</b><br>
# For an overview of Chronos-2 capabilities (univariate, multivariate, covariates), see the <a href="https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb"><b>Chronos-2 Quick Start notebook</b></a>.
# </div>

# %% [markdown]
# <div class="alert alert-warning">
# <b>⚠️ Looking for Chronos-Bolt or original Chronos?</b><br>
# This notebook covers <b>Chronos-2</b>, the latest and recommended model. For documentation on older models (Chronos-Bolt and original Chronos), see the <a href="https://github.com/amazon-science/chronos-forecasting/blob/v1.5.3/notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb"><b>legacy deployment walkthrough</b></a>.
# </div>

# %% [markdown]
# ## Setup

# %%
# !pip install -U -q "sagemaker<3"

# %% [markdown]
# If running in a SageMaker Notebook with the correct execution role, `role` can be set to `None`. Otherwise, specify your IAM role ARN.

# %%
role = None  # or "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXXXXXXXXXXX"

# %% [markdown]
# ---
# ## Section 1: Real-time Inference
#
# Real-time inference is the simplest option. SageMaker keeps a dedicated instance running, ready to serve predictions with low latency.
#
# **When to use:**
# - Interactive applications that need sub-second response times
# - Consistent, predictable traffic
# - When simplicity matters more than cost optimization

# %% [markdown]
# ### Deploy the Model
#
# With SageMaker JumpStart, you configure the deployment with a few parameters:
#
# - `model_id`: The model to deploy. Use `pytorch-forecasting-chronos-2` for [Chronos-2](https://huggingface.co/amazon/chronos-2).
# - `instance_type`: The AWS instance type for serving. Supported options:
#   - **GPU**: `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g6e.xlarge`, `ml.g6e.2xlarge`, `ml.g4dn.xlarge`, `ml.g4dn.2xlarge`
#   - **CPU**: `ml.m5.xlarge`, `ml.m5.2xlarge`, `ml.m5.4xlarge`, `ml.c5.xlarge`, `ml.c5.2xlarge`, `ml.c5.4xlarge`
#
# JumpStart automatically sets other attributes like `image_uri` based on your choices. See [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/) for instance costs.

# %%
from sagemaker.jumpstart.model import JumpStartModel

js_model = JumpStartModel(
    model_id="pytorch-forecasting-chronos-2",
    instance_type="ml.g5.2xlarge",
    role=role,
)

predictor = js_model.deploy()

# %% [markdown]
# > **Note:** After the endpoint is deployed, it will incur charges until you delete it with `predictor.delete_predictor()`

# %% [markdown]
# To connect to an existing endpoint instead:

# %%
# from sagemaker.predictor import Predictor
# from sagemaker.serializers import JSONSerializer
# from sagemaker.deserializers import JSONDeserializer
#
# predictor = Predictor("NAME_OF_EXISTING_ENDPOINT", serializer=JSONSerializer(), deserializer=JSONDeserializer())

# %% [markdown]
# ### Query the Endpoint

# %%
from pprint import pformat


def nested_round(data, decimals=2):
    """Round numbers, including nested dicts and lists."""
    if isinstance(data, float):
        return round(data, decimals)
    elif isinstance(data, list):
        return [nested_round(item, decimals) for item in data]
    elif isinstance(data, dict):
        return {key: nested_round(value, decimals) for key, value in data.items()}
    return data


def pretty_format(data):
    return pformat(nested_round(data), width=150, sort_dicts=False)

# %% [markdown]
# #### Univariate Forecasting

# %%
payload = {
    "inputs": [
        {"target": [0.0, 4.0, 5.0, 1.5, -3.0, -5.0, -3.0, 1.5, 5.0, 4.0, 0.0, -4.0, -5.0, -1.5, 3.0, 5.0, 3.0, -1.5, -5.0, -4.0]},
    ],
    "parameters": {"prediction_length": 10},
}
response = predictor.predict(payload)
print(pretty_format(response))

# %% [markdown]
# #### Multiple Time Series with Metadata

# %%
payload = {
    "inputs": [
        {"target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0], "item_id": "product_A", "start": "2024-01-01T01:00:00"},
        {"target": [5.4, 3.0, 3.0, 2.0, 1.5, 2.0, -1.0], "item_id": "product_B", "start": "2024-02-02T03:00:00"},
    ],
    "parameters": {"prediction_length": 5, "freq": "1h", "quantile_levels": [0.1, 0.5, 0.9]},
}
response = predictor.predict(payload)
print(pretty_format(response))

# %% [markdown]
# #### Forecasting with Covariates

# %%
payload = {
    "inputs": [
        {
            "target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0],
            "past_covariates": {
                "feat_1": [3.0, 6.0, 9.0, 6.0, 1.5, 6.0, 9.0, 6.0, 3.0],
                "feat_2": ["A", "B", "B", "B", "A", "A", "A", "A", "B"],
                "feat_3": [10.0, 20.0, 30.0, 20.0, 5.0, 20.0, 30.0, 20.0, 10.0],  # past-only
            },
            "future_covariates": {"feat_1": [2.5, 2.2, 3.3], "feat_2": ["B", "A", "A"]},
        },
    ],
    "parameters": {"prediction_length": 3, "quantile_levels": [0.1, 0.5, 0.9]},
}
response = predictor.predict(payload)
print(pretty_format(response))

# %% [markdown]
# #### Multivariate Forecasting

# %%
payload = {
    "inputs": [
        {
            "target": [
                [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0],  # Dimension 1
                [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],  # Dimension 2
                [2.0, 2.5, 3.0, 2.5, 2.0, 2.5, 3.0, 3.5],  # Dimension 3
            ],
        },
    ],
    "parameters": {"prediction_length": 4, "quantile_levels": [0.1, 0.5, 0.9]},
}
response = predictor.predict(payload)
print(pretty_format(response))

# %% [markdown]
# ### Working with Long-Format DataFrames
#
# Time series data is often stored in long-format DataFrames. The following helper functions convert between DataFrame and payload formats. You can skip this section if you prefer to construct payloads manually.

# %%
import pandas as pd


def convert_df_to_payload(
    past_df,
    future_df=None,
    prediction_length=1,
    freq="D",
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
):
    """
    Converts past and future DataFrames into JSON payload format for the Chronos endpoint.

    Args:
        past_df: Historical data with target, timestamp_column, and id_column.
        future_df: Future covariates with timestamp_column and id_column.
        prediction_length: Number of future time steps to predict.
        freq: Pandas-compatible frequency of the time series.
        target: Column name(s) for target values (str for univariate, list for multivariate).
        id_column: Column name for item IDs.
        timestamp_column: Column name for timestamps.

    Returns:
        dict: JSON payload formatted for the Chronos endpoint.
    """
    past_df = past_df.sort_values([id_column, timestamp_column])
    if future_df is not None:
        future_df = future_df.sort_values([id_column, timestamp_column])

    target_cols = [target] if isinstance(target, str) else target
    past_covariate_cols = list(past_df.columns.drop([*target_cols, id_column, timestamp_column]))
    future_covariate_cols = [] if future_df is None else [col for col in past_covariate_cols if col in future_df.columns]

    inputs = []
    for item_id, past_group in past_df.groupby(id_column):
        if len(target_cols) > 1:
            target_values = [past_group[col].tolist() for col in target_cols]
            series_length = len(target_values[0])
        else:
            target_values = past_group[target_cols[0]].tolist()
            series_length = len(target_values)

        if series_length < 5:
            raise ValueError(f"Time series '{item_id}' has fewer than 5 observations.")

        series_dict = {
            "target": target_values,
            "item_id": str(item_id),
            "start": past_group[timestamp_column].iloc[0].isoformat(),
        }

        if past_covariate_cols:
            series_dict["past_covariates"] = past_group[past_covariate_cols].to_dict(orient="list")

        if future_covariate_cols:
            future_group = future_df[future_df[id_column] == item_id]
            if len(future_group) != prediction_length:
                raise ValueError(
                    f"future_df must contain exactly {prediction_length=} values for each item_id from past_df "
                    f"(got {len(future_group)=}) for {item_id=}"
                )
            series_dict["future_covariates"] = future_group[future_covariate_cols].to_dict(orient="list")

        inputs.append(series_dict)

    return {
        "inputs": inputs,
        "parameters": {"prediction_length": prediction_length, "freq": freq},
    }


def convert_response_to_df(response, freq="D"):
    """
    Converts a JSON response from the Chronos endpoint into a long-format DataFrame.

    Args:
        response: JSON response containing forecasts.
        freq: Pandas-compatible frequency of the time series.

    Returns:
        pd.DataFrame: Long-format DataFrame with timestamps, item_id, and forecasted values.
    """
    dfs = []
    for forecast in response["predictions"]:
        if isinstance(forecast["mean"], list) and isinstance(forecast["mean"][0], list):
            # Multivariate forecast
            timestamps = pd.date_range(forecast["start"], freq=freq, periods=len(forecast["mean"][0]))
            for dim_idx in range(len(forecast["mean"])):
                dim_data = {"item_id": forecast.get("item_id"), "timestamp": timestamps, "target": f"target_{dim_idx + 1}"}
                for key, value in forecast.items():
                    if key not in ["item_id", "start"]:
                        dim_data[key] = value[dim_idx]
                dfs.append(pd.DataFrame(dim_data))
        else:
            # Univariate forecast
            forecast_df = pd.DataFrame(forecast).drop(columns=["start"])
            forecast_df["timestamp"] = pd.date_range(forecast["start"], freq=freq, periods=len(forecast_df))
            cols = ["item_id", "timestamp"] + [c for c in forecast_df.columns if c not in ["item_id", "timestamp"]]
            forecast_df = forecast_df[cols]
            dfs.append(forecast_df)

    return pd.concat(dfs, ignore_index=True)

# %%
df = pd.read_csv(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    parse_dates=["timestamp"],
)

prediction_length = 8
target_col = "unit_sales"
freq = pd.infer_freq(df[df.item_id == df.item_id[0]]["timestamp"])

past_df = df.groupby("item_id").head(-prediction_length)
future_df = df.groupby("item_id").tail(prediction_length).drop(columns=[target_col])
# %%
past_df.head()
# %%
future_df.head()
# %%
payload = convert_df_to_payload(past_df, future_df, prediction_length=prediction_length, freq=freq, target="unit_sales")
response = predictor.predict(payload)
forecast_df = convert_response_to_df(response, freq=freq)
forecast_df.head()

# %% [markdown]
# ### Clean Up
#
# The endpoint incurs charges until deleted. Alternatively, you can configure [scaling to zero](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html) to save costs when the endpoint is idle.

# %%
predictor.delete_predictor()

# %% [markdown]
# ---
# ## Setup for Serverless Inference and Batch Transform
#
# Serverless Inference and Batch Transform only support CPU instances. Unlike real-time inference with JumpStart, these modes require you to create a custom SageMaker Model with repackaged artifacts.
#
# The following section sets up a reusable model that you can use for both Serverless (Section 2) and Batch Transform (Section 3).

# %%
import boto3
import json
import tempfile
import tarfile
from pathlib import Path
from sagemaker import Session
from sagemaker.model import Model
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def repackage_jumpstart_model(js_model, output_bucket, output_key):
    """
    Repackages JumpStart model artifacts into a single tar.gz file for serverless/batch deployment.

    Args:
        js_model: JumpStartModel instance with model_data configured.
        output_bucket: S3 bucket to store the repackaged model.
        output_key: S3 key for the output tar.gz file.

    Returns:
        str: S3 URI of the repackaged model.
    """
    s3 = boto3.client("s3")
    s3_uri = js_model.model_data["S3DataSource"]["S3Uri"].rstrip("/") + "/"
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download all model artifacts
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/"):
                    local_file = tmpdir / obj["Key"][len(prefix):]
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, obj["Key"], str(local_file))

        # Create tar.gz archive
        tar_path = tmpdir / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(tmpdir, arcname=".")

        s3.upload_file(str(tar_path), output_bucket, output_key)

    return f"s3://{output_bucket}/{output_key}"

# %% [markdown]
# ### Create the SageMaker Model
#
# This model can be used for both Serverless Inference and Batch Transform.

# %%
# Reuse the role defined in Setup, or define a new one
# role = None  # or "arn:aws:iam::..."

# Use JumpStart to get the model artifacts and container image
js_model = JumpStartModel(
    model_id="pytorch-forecasting-chronos-2",
    instance_type="ml.c5.4xlarge",  # Important: use CPU instance to ensure that correct image_uri is used
    role=role,
)

# Repackage model artifacts into a single tar.gz
session = Session()
bucket = session.default_bucket()  # or "your-bucket-name"
s3_prefix = "chronos-2"  # S3 prefix for model artifacts and data

model_uri = repackage_jumpstart_model(js_model, bucket, output_key=f"{s3_prefix}/model.tar.gz")
print(f"Repackaged model uploaded to: {model_uri}")

# %%
from sagemaker.predictor import Predictor

chronos_model = Model(
    name="chronos-2-cpu",  # Important: Model name should start with 'chronos-2'
    model_data=model_uri,
    image_uri=js_model.image_uri,
    role=role,
    predictor_cls=Predictor,
)
chronos_model.create()

# %% [markdown]
# Alternatively, you can load an existing model as follows:

# %%
# model_info = boto3.client("sagemaker").describe_model(ModelName="chronos-2-cpu")
# model = Model(
#     model_data=model_info["PrimaryContainer"]["ModelDataUrl"],
#     image_uri=model_info["PrimaryContainer"]["Image"],
#     role=model_info["ExecutionRoleArn"],
#     name=model_info["ModelName"],
# )

# %% [markdown]
# ---
# ## Section 2: Serverless Inference
#
# [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html) scales compute capacity based on traffic and scales to zero when idle, so you only pay for actual inference time.
#
# **When to use:**
# - Sporadic or unpredictable traffic
# - Cost-sensitive workloads with variable demand
# - Development and testing environments
#
# **Limitations:**
# - Cold start latency (first request after idle typically takes 30-60 seconds)
# - Maximum memory: 6GB

# %% [markdown]
# ### Deploy Serverless Endpoint

# %%
from sagemaker.serverless import ServerlessInferenceConfig

serverless_predictor = chronos_model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=6144,  # Maximum available memory
        max_concurrency=1,
    ),
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

# %% [markdown]
# ### Query Serverless Endpoint
# %%
payload = {
    "inputs": [
        {"target": [0.0, 4.0, 5.0, 1.5, -3.0, -5.0, -3.0, 1.5, 5.0, 4.0, 0.0, -4.0, -5.0, -1.5, 3.0, 5.0, 3.0, -1.5, -5.0, -4.0]},
    ],
    "parameters": {"prediction_length": 10},
}
response = serverless_predictor.predict(payload)
print(pretty_format(response))

# %% [markdown]
# ### Clean Up

# %%
serverless_predictor.delete_predictor()

# %% [markdown]
# ---
# ## Section 3: Batch Transform
#
# [Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html) processes large datasets offline. SageMaker spins up compute, processes all data, and shuts down automatically.
#
#
# **When to use:**
# - Large-scale batch forecasting (thousands of time series)
# - Scheduled or periodic forecasting jobs
# - When latency is not critical
#
# **Limitations:**
# - Not suitable for real-time predictions
# - Requires data to be staged in S3

# %% [markdown]
# ### Prepare Input Data
#
# The model uses the same API as described in the Endpoint API Reference at the end of the notebook, so you need to prepare your data in the expected JSON format.
#
# Batch Transform reads input from S3. Each line in the input file is a JSON payload that can contain multiple time series. For large datasets, use `items_per_record` to control how many time series are included per line (and thus per request).

# %%
# Load sample data
df = pd.read_csv(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    parse_dates=["timestamp"],
)

prediction_length = 8
target_col = "unit_sales"
freq = pd.infer_freq(df[df.item_id == df.item_id[0]]["timestamp"])

past_df = df.groupby("item_id").head(-prediction_length)
future_df = df.groupby("item_id").tail(prediction_length).drop(columns=[target_col])

# Convert DataFrame to payload and split into chunks
payload = convert_df_to_payload(past_df, future_df, prediction_length=prediction_length, freq=freq, target=target_col)
items_per_record = 100  # Number of time series per JSONL line
inputs, params = payload["inputs"], payload["parameters"]
lines = [json.dumps({"inputs": inputs[i:i + items_per_record], "parameters": params}) for i in range(0, len(inputs), items_per_record)]

# Upload input data to S3
input_key = f"{s3_prefix}/batch-input/input.jsonl"
boto3.client("s3").put_object(Bucket=bucket, Key=input_key, Body="\n".join(lines).encode())
input_s3_uri = f"s3://{bucket}/{input_key}"
print(f"Input data uploaded to: {input_s3_uri} ({len(lines)} records)")

# %% [markdown]
# ### Run Batch Transform
#
# This uses the same `chronos_model` created in the setup section above.

# %%
from sagemaker.transformer import Transformer

output_s3_uri = f"s3://{bucket}/{s3_prefix}/batch-output/"

transformer = Transformer(
    model_name=chronos_model.name,
    instance_count=1,
    instance_type="ml.c5.4xlarge",  # CPU instance
    output_path=output_s3_uri,
    strategy="SingleRecord",  # Process one JSON line at a time
    assemble_with="Line",
    accept="application/json",
)

transformer.transform(
    data=input_s3_uri,
    content_type="application/json",
    split_type="Line",
    wait=True,
)

# %% [markdown]
# ### Retrieve Batch Results

# %%
output_key = f"{s3_prefix}/batch-output/{input_key.split('/')[-1]}.out"
result = boto3.client("s3").get_object(Bucket=bucket, Key=output_key)
output_lines = result["Body"].read().decode().strip().split("\n")

# Combine predictions from all records
all_predictions = [p for line in output_lines for p in json.loads(line)["predictions"]]
forecast_df = convert_response_to_df({"predictions": all_predictions}, freq=freq)
forecast_df.head()

# %% [markdown]
# ---
# ## See Also
#
# - [Scale real-time endpoints to zero](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html) to optimize costs when the endpoint is idle
# - [Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) handles traffic spikes better than real-time inference thanks to request queueing

# %% [markdown]
# ---
# ## Endpoint API Reference
#
# Below is a complete API specification for the Chronos-2 endpoint.
#
# * **inputs** (required): List with at most 1000 time series that need to be forecasted. Each time series is represented by a dictionary with the following keys:
#     * **target** (required): Observed time series values.
#         - For univariate forecasting: List of numeric values.
#         - For multivariate forecasting: List of lists, where each inner list represents one dimension. All dimensions must have the same length. If converted to a numpy array via `np.array(target)`, the shape would be `[num_dimensions, length]`.
#         - It is recommended that each time series contains at least 30 observations.
#         - If any time series contains fewer than 5 observations, an error will be raised.
#     * **item_id**: String that uniquely identifies each time series.
#         - If provided, the ID must be unique for each time series.
#         - If provided, then the endpoint response will also include the **item_id** field for each forecast.
#     * **start**: Timestamp of the first time series observation in ISO format (`YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ss`).
#         - If **start** field is provided, then **freq** must also be provided as part of **parameters**.
#         - If provided, then the endpoint response will also include the **start** field indicating the first timestamp of each forecast.
#     * **past_covariates**: Dictionary containing the past values of the covariates for this time series.
#         - Each key in **past_covariates** corresponds to the name of the covariate. Each value must be an array consisting of all-numeric or all-string values, with the length equal to the length of the **target**.
#         - Covariates that appear only in **past_covariates** (and not in **future_covariates**) are treated as past-only covariates.
#     * **future_covariates**: Dictionary containing the future values of the covariates for this time series (values during the forecast horizon).
#         - Each key in **future_covariates** corresponds to the name of the covariate. Each value must be an array consisting of all-numeric or all-string values, with the length equal to **prediction_length**.
#         - Covariates that appear in both **past_covariates** and **future_covariates** are treated as known future covariates.
# * **parameters**: Optional parameters to configure the model.
#     * **prediction_length**: Integer corresponding to the number of future time series values that need to be predicted. Defaults to `1`. Values up to `1024` are supported.
#     * **quantile_levels**: List of floats in range (0, 1) specifying which quantiles should be included in the probabilistic forecast. Defaults to `[0.1, 0.5, 0.9]`.
#         - Chronos-2 natively supports quantile levels in range `[0.01, 0.99]`. Predictions outside the range will be clipped.
#     * **freq**: Frequency of the time series observations in [pandas-compatible format](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases). For example, `1h` for hourly data or `2W` for bi-weekly data.
#         - If **freq** is provided, then **start** must also be provided for each time series in **inputs**.
#     * **batch_size**: Number of time series processed in parallel by the model. Larger values speed up inference but may lead to out of memory errors. Defaults to `256`.
#     * **cross_learning**: If `True`, the model will apply group attention to all items in the batch, instead of processing each item separately (described as "full cross-learning mode" in the [technical report](https://www.arxiv.org/abs/2510.15821)). This may produce more accurate forecasts for some tasks. Defaults to `False`.
#
# All keys not marked with (required) are optional.
#
# The endpoint response contains the probabilistic (quantile) forecast for each time series included in the request.
