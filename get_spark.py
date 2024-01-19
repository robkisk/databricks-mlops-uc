import os


class GetSpark:
    def __init__(self, profile: str = "DEFAULT"):
        self.profile = profile

    def is_running_in_databricks(self) -> bool:
        return "DATABRICKS_RUNTIME_VERSION" in os.environ

    def init_spark(self, eager=True):
        if self.is_running_in_databricks():
            from pyspark.sql import SparkSession

            return SparkSession.builder.getOrCreate()

        from databricks.connect import DatabricksSession
        from databricks.sdk.core import Config

        config = Config(profile=self.profile)
        spark = DatabricksSession.builder.sdkConfig(config).getOrCreate()
        if eager:
            spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
            spark.conf.set("spark.sql.repl.eagerEval.maxNumRows", 5)
            return spark
        return spark
