# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC > To demo the following, you can simulate cross-workspace sharing of models using the following field-eng workspaces:
# MAGIC > -  Dev/Staging workspace: https://e2-demo-west.cloud.databricks.com
# MAGIC > -  Prod workspace: https://e2-demo-field-eng.cloud.databricks.com
# MAGIC >
# MAGIC > Run this notebook in the [Prod workspace](https://e2-demo-field-eng.cloud.databricks.com) to simulate running the model training pipeline. Following this, demonstrate the ability to load the trained model in the [Dev workspace](https://e2-demo-field-eng.cloud.databricks.com) from the `prod` catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Models in Unity Catalog - NYC Taxi Trip Duration Model
# MAGIC
# MAGIC This is the base notebook of our project and is used to demonstrate a simple model training pipeline, where we predict the duration of [taxi trips in New York City](https://www.kaggle.com/c/nyc-taxi-trip-duration). We started our project with this notebook and proceeded to refactor and modularize it into a python package to be deployed in an end to end MLOps workflow.
# MAGIC
# MAGIC The core aim of this notebook is to demonstrate how to register a model to Unity Catalog, and subsequently load the model for inference. The wider repo in which this notebook sits aims to demonstrate how to go from this notebook to a productionized ML application, using Unity Catalog to manage our registered model.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC The following requirements are needed in order to be able to register ML models in Unity Catalog:
# MAGIC
# MAGIC * A cluster running Databricks Runtime 13.0 or above with access to Unity Catalog ([AWS](https://docs.databricks.com/data-governance/unity-catalog/compute.html#create-clusters--sql-warehouses-with-unity-catalog-access)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/compute))
# MAGIC * Permissions to create models in at least one Unity Catalog schema. In particular, you need `USE CATALOG` permissions on the parent catalog, and both `USE SCHEMA` and `CREATE MODEL` permissions on the parent schema. If you hit permissions errors while running the example below, ask a catalog/schema owner or admin for access.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The notebook contains the following steps:
# MAGIC
# MAGIC 1. **Imports**: Necessary libraries and modules are imported. MLflow autologging is enabled and the registry URI is set to "databricks-uc".
# MAGIC 1. **Global Variables**: Set global variables to be used throughout the notebook.
# MAGIC 1. **Load Data**: Load the NYC Taxi Trip Duration dataset.
# MAGIC 1. **Split Data**: The loaded data is split into training, validation, and test sets.
# MAGIC 1. **Feature Engineering**: The input DataFrame is extended with additional features, and unneeded columns are dropped. Define a Scikit-Learn pipeline to perform feature engineering.
# MAGIC 1. **Train Model**: Train an XGBoost Regressor model, tracking parameters, metrics and model artifacts to MLflow.
# MAGIC 1. **Register Model**: The trained model is registered to Unity Catalog. Update the registered model with a "Champion" alias.
# MAGIC 1. **Consume Model**: The "Champion" version of the registered model is loaded and used for inference against the test dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports
# MAGIC
# MAGIC Import all necessary libraries and modules required for the notebook.
# MAGIC Note that we also enable MLflow autologging and define the catalog and schema name to which we will be registering our model.
# MAGIC
# MAGIC By default, this example creates models under the `main` catalog and `default` schema in Unity Catalog. You can optionally also specify a different catalog or schema on which you have the necessary permissions.
# MAGIC
# MAGIC To create a new catalog or schema, you can use `CREATE CATALOG` ([AWS](https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-catalog.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-catalog)) or `CREATE SCHEMA` ([AWS](https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-schema.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-schema)), respectively

# COMMAND ----------

# MAGIC %md
# MAGIC Install version 2.3.2 or above of the MLflow Python client with databricks extras on your cluster

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=2.3.0"

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Turn on MLflow autologging
# Note that autologging is on by default in Databricks notebooks.
# See https://mlflow.org/docs/latest/tracking.html#automatic-logging for more details.
mlflow.autolog()

# Set the registry URI to "databricks-uc" to configure the MLflow client to access models in UC
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Global Variables
# MAGIC
# MAGIC Set global variables to be used throughout the notebook.

# COMMAND ----------
EXPERIMENT_PATH = "/Users/robby.kiskanyan@databricks.com/models_in_uc/nyc_taxi_duration"
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------
# UC registered model name
# A UC registered_model_name follows the pattern <catalog_name>.<schema_name>.<model_name>,
# corresponding to the catalog, schema, and registered model name
# in Unity Catalog under which to create the model version.
CATALOG_NAME = "robkisk"
SCHEMA_NAME = "ml"
MODEL_NAME = "nyc_taxi_duration_model_robkisk"
REGISTERED_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load data from Delta
# MAGIC
# MAGIC Load the NYC Taxi Trip Duration dataset from Delta. The resulting data is converted to a pandas DataFrame.

# COMMAND ----------


def load_data_delta(filepath: str) -> pd.DataFrame:
    """Load NYC Taxi Trip Duration dataset from Delta.

    Args:
        filepath (str): Path to the Delta directory.

    Returns:
        pd.DataFrame: Loaded data.
    """
    sdf = spark.read.format("delta").load(filepath)
    pdf = sdf.toPandas()
    return pdf


# COMMAND ----------

# Load dataset from Delta
nyc_taxi_pdf = load_data_delta(
    filepath="dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled"
)
nyc_taxi_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Split data
# MAGIC
# MAGIC Split the dataset into train, validation and test sets prior to model training. We set a default split ratio of 70% for training, 20% for validation, and 10% for testing.

# COMMAND ----------


def split_data(
    pdf: pd.DataFrame, target_column: str, split_ratio: tuple = (0.7, 0.2, 0.1)
) -> tuple:
    """Split the data into a training set, validation set, and a test set.

    Args:
        pdf (pd.DataFrame): Input data.
        target_column (str): Name of the target column.
        split_ratio (tuple): A tuple that specifies the ratio of the training, validation, and test sets.

    Returns:
        tuple: A tuple containing the features and target for the training, validation, and test sets.
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1"

    X = pdf.drop(target_column, axis=1)
    y = pdf[target_column]

    # Calculate split sizes
    train_size, val_size = split_ratio[0], split_ratio[0] + split_ratio[1]

    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=split_ratio[2], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (train_size + val_size),
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature engineering
# MAGIC
# MAGIC We create the following functions for feature engineering: `calculate_features()` and `transformer_fn()`.
# MAGIC
# MAGIC - `calculate_features()`: used to extend the input DataFrame with pickup day of the week and hour, and trip duration.
# MAGIC - `transformer_fn()`: returns an unfitted transformer that defines `fit()` and `transform()` methods which perform feature engineering and encoding.
# MAGIC
# MAGIC We will use the resulting `transformer_fn()` as part of our sklearn `Pipeline`.

# COMMAND ----------


def calculate_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Function to conduct feature engineering.

    Extend the input dataframe with pickup day of week and hour, and trip duration.
    Drop the now-unneeded pickup datetime and dropoff datetime columns.

    Args:
        pdf (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    pdf["pickup_dow"] = pdf["tpep_pickup_datetime"].dt.dayofweek
    pdf["pickup_hour"] = pdf["tpep_pickup_datetime"].dt.hour
    trip_duration = pdf["tpep_dropoff_datetime"] - pdf["tpep_pickup_datetime"]
    pdf["trip_duration"] = trip_duration.map(lambda x: x.total_seconds() / 60)
    pdf.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)
    return pdf


def transformer_fn() -> Pipeline:
    """Define sklearn pipeline.

    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted sklearn transformer
    """
    return Pipeline(
        steps=[
            (
                "calculate_time_and_duration_features",
                FunctionTransformer(calculate_features, feature_names_out="one-to-one"),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "hour_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["pickup_hour"],
                        ),
                        (
                            "day_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["pickup_dow"],
                        ),
                        (
                            "std_scaler",
                            StandardScaler(),
                            ["trip_distance", "trip_duration"],
                        ),
                    ]
                ),
            ),
        ]
    )


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Train model
# MAGIC
# MAGIC ML model versions in UC must have a model signature. If you’re not already logging MLflow models with signatures in your model training workloads, you can either:
# MAGIC
# MAGIC 1. Use MLflow autologging
# MAGIC     - MLflow autologing automatically logs models when they are trained in a notebook. Model signature is inferred and logged alongside the model artifacts.
# MAGIC    - Read https://mlflow.org/docs/latest/tracking.html#automatic-logging to see if your model flavor is supported.
# MAGIC 2. Manually set the model signature in `mlflow.<flavor>.log_model`
# MAGIC     - Infer model signature via [`mlflow.models.infer_signature`](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.infer_signature), or manually define the signature.
# MAGIC     - Pass the model signature to `log_model` via the `signature` argument
# MAGIC
# MAGIC Given that we have enabled MLflow autologging at the outset of the notebook we will not need to explcitly set the model signature.
# MAGIC
# MAGIC In the following cell:
# MAGIC
# MAGIC - `estimator_fn()` defines an unfitted `XGBRegressor` estimator that defines (using the sklearn API). This is subsequently used as the estimator in our sklearn `Pipeline`.
# MAGIC - `train_model` creates and fits our sklearn `Pipeline`, tracking to MLflow.

# COMMAND ----------


def estimator_fn(*args, **kwargs) -> BaseEstimator:
    """Define XGBRegressor model.

    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.

    Returns:
        sklearn.base.BaseEstimator: Unfitted sklearn base estimator
    """
    return XGBRegressor(objective="reg:squarederror", random_state=42, *args, **kwargs)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Function to trigger model training, tracking to MLflow Tracking.

    Create a pipeline that includes feature engineering and model training, and fit it on the training data.
    Return the run_id of the MLflow run.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.

    Returns:
        str: MLflow run_id.
    """
    with mlflow.start_run():
        pipeline = Pipeline(
            steps=[("transformer", transformer_fn()), ("model", estimator_fn())]
        )

        pipeline.fit(X_train, y_train)

        return mlflow.active_run().info.run_id


# COMMAND ----------

# Split data into train/val/test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    nyc_taxi_pdf, target_column="fare_amount", split_ratio=(0.7, 0.2, 0.1)
)
# COMMAND ----------
# Trigger model training
run_id = train_model(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Register model
# MAGIC
# MAGIC Here we register the ML model version to Unity Catalog. This is almost identical to how we'd use the workspace Model Registry, but with a few differences:
# MAGIC
# MAGIC 1. We configure the MLflow client to register models to Unity Catalog by setting the registry URI to `databricks-uc`
# MAGIC 2. Registered models in Unity Catalog are specified via their full three-level name (`<catalog_name>.<schema_name>.<model_name>`), so we update the model name we pass to MLflow APIs accordingly

# COMMAND ----------
# COMMAND ----------

print(f"REGISTERED_MODEL_NAME: {REGISTERED_MODEL_NAME}")

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", name=REGISTERED_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md ## View ML models in the UI
# MAGIC
# MAGIC You can view ML models in UC in the [Data Explorer](./explore/data?filteredObjectTypes=REGISTERED_MODEL), under the catalog and schema in which the model was created.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Mark model for deployment using aliases
# MAGIC
# MAGIC Model aliases help replace model registry stages in the UC Model Registry, allowing you to assign a mutable, named reference to a particular version within a registered model.
# MAGIC You can use aliases to specify which model versions are deployed in a given environment in your model training workflows (e.g. specify the current "Champion" model version that should serve the majority of production traffic), and then write inference workloads that target that alias (“make predictions using the ‘Champion’ version”). The example workflow below captures this idea.
# MAGIC
# MAGIC In this example, we have already registered our ML model to Unity Catalog to our `<CATALOG_NAME>` catalog. We can additionally update the `"Champion"` alias to point to our newly trained model version, indicating to downstream inference workloads that they should use this model version to make predictions on production traffic.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME, alias="Champion", version=model_version.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Consume model versions by alias in inference workloads
# MAGIC
# MAGIC We can reference our model version by alias at inference time. Below we load and use the `"Champion"` model version for inference on our test set. If the `"Champion"` version is updated to reference a new model version, this model would be automatically loaded and used on the next execution. This allows you to decouple model deployments/updates from downstream batch inference pipelines.

# COMMAND ----------

import mlflow.pyfunc

model_uri = f"models:/{REGISTERED_MODEL_NAME}@Champion"
print(f"model_uri: {model_uri}")

champion_model = mlflow.pyfunc.load_model(model_uri)
champion_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Share models across workspaces
# MAGIC
# MAGIC As long as you have the appropriate permissions, you can access models in UC from any workspace that you have access to. For example, Data Scientists developing models in a "dev" workspace may lack permissions in a "prod" workspace. Using models in Unity Catalog, they would be able to access models trained in the "prod" workspaces - and registered to the "prod" catalog - from the "dev" workspace. Thus enabling those Data Scientists to compare newly-developed models to the production baseline.
# MAGIC
# MAGIC If you’re not ready to move full production model training or inference pipelines to the private preview UC model registry, you can still leverage UC for cross-workspace model sharing, by registering new model versions to both Unity Catalog and workspace model registries.
