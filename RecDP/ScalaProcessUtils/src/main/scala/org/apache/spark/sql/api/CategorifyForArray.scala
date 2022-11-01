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

package org.apache.spark.sql.api

import scala.collection.mutable.WrappedArray

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object CategorifyForArray {
  def categorify(sc: JavaSparkContext, df: DataFrame, col_name: String, dict_df: DataFrame): DataFrame = {
    val broadcast_data = dict_df.select("dict_col", "dict_col_id").collect().map( row => 
      (row(0).asInstanceOf[String], row(1).asInstanceOf[Int])
    ).toMap
    val broadcast_handler = sc.broadcast(broadcast_data)
    val categorifyUDF = udf((x_l: WrappedArray[String]) => {
      val broadcasted = broadcast_handler.value
      if (broadcasted == null || broadcasted.isEmpty || x_l == null) {
        Array[Int]()
      } else {
        x_l.toArray.map(x => if (x != null && broadcasted.contains(x)) broadcasted(x) else 0)
      }
    })
    return df.withColumn(col_name, categorifyUDF(col(col_name)))
  }
}