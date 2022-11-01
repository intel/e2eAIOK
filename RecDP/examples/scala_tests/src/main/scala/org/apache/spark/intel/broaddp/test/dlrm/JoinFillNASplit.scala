/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package  org.apache.spark.broaddp.test.dlrm

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.SparkSession
import java.util.UUID.randomUUID

object JoinFillNASplit {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("yarn").appName("DLRM_JOIN_FILLNA_SCALA").getOrCreate()
    val catCols = (14 until 40 toArray).map(i => "_c" + i)
    val intCols = (1 until 14 toArray).map(i => "_c" + i)

    val path_prefix = "hdfs://"
    val parquet_folder = "/dlrm/parquet_raw_data/"
    val parquet_path = s"$path_prefix/$parquet_folder"
    val files = (0 until 24 toList).map(i => s"$parquet_folder/day_$i")
    val models_folder = "/dlrm/models/"

    val start = System.nanoTime
    var df = spark.read.parquet(files:_*)

    for( i <- 14 to 39) {
      val colName = s"_c$i"
      val model = spark.read.parquet(s"$models_folder/_c$i").withColumn("id", row_number().over(
        Window.orderBy(desc("model_count")))).withColumn("id", col("id") - 1).select(col("data").alias(colName), col("id"))
      val broadcastModel = broadcast(model)
      df = df.join(broadcastModel, df(colName) === broadcastModel(colName), joinType="left")
        .drop(colName)
        .withColumnRenamed("id", colName)
    }

    df.printSchema
    df = df.repartition(2000)
    df = df.na.fill(0, cols = intCols ++ catCols)
    val saved_path = s"/dlrm/tmp/${randomUUID().toString}/"
    df.write.format("parquet").mode("overwrite").save(saved_path)
    
    val end = System.nanoTime
    println("Total time: " + (end - start) / 1e9d)
    println("Saved to: " + saved_path)
    spark.stop()
  }
} 
