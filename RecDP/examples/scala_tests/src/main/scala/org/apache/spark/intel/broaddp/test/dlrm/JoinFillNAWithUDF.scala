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

package org.apache.spark.broaddp.test.dlrm

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import java.util.UUID.randomUUID
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.collection.OpenHashMap


object JoinFillNAWithUDF {
  var spark: SparkSession = _

  def categorify_udf(df: DataFrame, dict_df: DataFrame, col_name: String): DataFrame = {
    val sorted_data = dict_df.orderBy(desc("model_count")).collect()
    // option 1, use OpenHashMap
    val broadcast_data = new OpenHashMap[String, Int](sorted_data.size)
    sorted_data.zipWithIndex.foreach { case (row, idx) =>
      broadcast_data.update(row(0).asInstanceOf[String], idx)
    }
    // option 2, use Map
    // val broadcast_data: Array[(String, Int)] = sorted_data.zipWithIndex.map{case (row, idx) => (row(0).asInstanceOf[String], idx)}.toMap

    val broadcast_handler = spark.sparkContext.broadcast(broadcast_data)

    def mapToIndex(x: String): Int = {
      val broadcasted = broadcast_handler.value
      if (broadcasted.contains(x)) {
          return broadcasted(x)
      } else {
          return -1
      }
    }

    val udf_mapToIndex = udf(mapToIndex(_))
    spark.udf.register("udf_mapToIndex", udf_mapToIndex.asNondeterministic())
    df.withColumn(col_name, udf_mapToIndex(col(col_name)))
  }

  def categorify(df: DataFrame, dict_dfs: Array[(String, DataFrame)]): DataFrame = {
    var new_df = df
    dict_dfs.foreach{case(col, dict_df) => 
      new_df = categorify_udf(new_df, dict_df, col)
    }
    new_df
  }

  def main(args: Array[String]): Unit = {
    spark = SparkSession.builder.master("yarn").appName("DLRM_JOIN_FILLNA_WITH_UDF_SCALA").getOrCreate()
    val catCols = (14 until 40 toArray).map(i => "_c" + i)
    val intCols = (1 until 14 toArray).map(i => "_c" + i)

    val path_prefix = "hdfs://"
    val parquet_folder = "/dlrm/parquet_raw_data/"
    val parquet_path = s"$path_prefix/$parquet_folder"
    val files = (0 until 24 toList).map(i => s"$parquet_folder/day_$i")
    val models_folder = "/dlrm/models/"

    val start = System.nanoTime
    var df = spark.read.parquet(files:_*)
    val dict_dfs = catCols.map(col => (col, spark.read.parquet(s"$models_folder/$col")))

    // categorify
    // option 1, using UDF
    df = categorify(df, dict_dfs)

    // FillNA
    df = df.na.fill(0, cols = intCols ++ catCols)

    // Save result to HDFS
    val saved_path = s"/dlrm/tmp/${randomUUID().toString}/"
    df.write.format("parquet").mode("overwrite").save(saved_path)
    // save to /dlrm/tmp/a5a0319c-e149-41db-ab20-29c8c3e88b47
    val end = System.nanoTime
    println("Total time: " + (end - start) / 1e9d)
    println("Saved to: " + saved_path)
    spark.stop()
  }
}
