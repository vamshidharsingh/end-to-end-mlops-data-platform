import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import psycopg2

def run_etl():
    spark = SparkSession.builder \
        .appName("CustomerETL") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.5.4") \
        .getOrCreate()
        
    input_path = "data/raw/customers.csv"
    output_path = "data/processed/features.parquet"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Data cleaning / Feature calculation
    df_clean = df.na.drop()
    df_clean = df_clean.withColumn("income_per_age", col("income") / col("age"))
    df_clean = df_clean.withColumn("active_user", when(col("website_visits") > 10, 1).otherwise(0))
    
    # Save to parquet
    df_clean.write.mode("overwrite").parquet(output_path)
    print(f"Data transformed and saved to {output_path}")
    
    # Save to PostgreSQL
    db_url = "jdbc:postgresql://postgres:5432/mlops_db"
    db_properties = {
        "user": "mlops",
        "password": "mlops",
        "driver": "org.postgresql.Driver"
    }
    try:
        df_clean.write.jdbc(url=db_url, table="customer_features", mode="overwrite", properties=db_properties)
        print("Data loaded to PostgreSQL successfully.")
    except Exception as e:
        print(f"Could not load to PostgreSQL (might be running locally without DB): {e}")

if __name__ == "__main__":
    run_etl()
