# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Inference

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The intention of the following notebook is to illustrate the ability to load a model trained in the prod workspace, and logged to the prod catalog frp, the dev workspace. Ensure you have trained and registered the model as outlined in the notebook `1_nyc_taxi_models_uc`.

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd

# Set the registry URI to "databricks-uc" to configure the MLflow client to access models in UC
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# UC registered model name
# A UC registered_model_name follows the pattern <catalog_name>.<schema_name>.<model_name>, 
# corresponding to the catalog, schema, and registered model name 
# in Unity Catalog under which to create the model version.
CATALOG_NAME = "niall_prod"                 # TODO: update with your <catalog_name>
SCHEMA_NAME = "ml"                          # TODO: update with your <schema_name>
MODEL_NAME = "nyc_taxi_duration_model"      # TODO: `update with your <model_name>
REGISTERED_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

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


# Load dataset from Delta
nyc_taxi_pdf = load_data_delta(filepath="dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled")

# COMMAND ----------

model_uri = f"models:/{REGISTERED_MODEL_NAME}@Champion"
print(f"model_uri: {model_uri}")

champion_model = mlflow.pyfunc.load_model(model_uri)
champion_model.predict(nyc_taxi_pdf)
