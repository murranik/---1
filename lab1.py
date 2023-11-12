import os
import string
import sys

from nltk import word_tokenize, ngrams
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, size, udf

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def preprocess_commit(message):
    message = message.translate(str.maketrans('', '', string.punctuation))
    message = message.lower()
    return message


def generate_3grams_from_text(text):
    tokens = word_tokenize(text)
    tokens = tokens[:5]
    if len(tokens) < 5:
        return []
    three_grams = list(ngrams(tokens, 3))
    return [" ".join(gram) for gram in three_grams]


spark = SparkSession.builder.appName("Lab1") \
    .config("spark.master", "local") \
    .config("spark.hadoop.fs", "local") \
    .getOrCreate()

spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

jsonl_file_path = "10K.github.jsonl"
df = spark.read.json(jsonl_file_path)

push_events = df.filter(size(col("payload.commits")) > 0)
push_events = push_events.filter(col("type") == "PushEvent")

push_events = push_events.select("payload.commits")

push_events = push_events.withColumn("commit_info", explode(col("commits")))

push_events = push_events.withColumn("author_name", col("commit_info.author.name"))
push_events = push_events.withColumn("commit_message", col("commit_info.message"))

preprocess_udf = udf(preprocess_commit, StringType())
push_events = push_events.withColumn("cleaned_message", preprocess_udf(col("commit_message")))

generate_3grams_udf = udf(generate_3grams_from_text, ArrayType(StringType()))

push_events = push_events.withColumn("3grams", generate_3grams_udf(col("cleaned_message")))

push_events = push_events.filter(size(col("3grams")) > 0)

push_events = push_events.withColumn("3gram1", col("3grams")[0])
push_events = push_events.withColumn("3gram2", col("3grams")[1])
push_events = push_events.withColumn("3gram3", col("3grams")[2])

push_events = push_events.select("author_name", "3gram1", "3gram2", "3gram3")

push_events.write.csv("output.csv", header=True, mode="overwrite")

spark.stop()
