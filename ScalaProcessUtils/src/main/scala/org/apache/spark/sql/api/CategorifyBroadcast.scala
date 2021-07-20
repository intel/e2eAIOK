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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType}

object CategorifyBroadcast {
  def broadcast(sc: JavaSparkContext, dict_df: DataFrame): Broadcast[Map[String, Integer]] = {
    val broadcast_data = dict_df.select("dict_col", "dict_col_id").collect().map( row => 
      (row(0).asInstanceOf[String], row(1).asInstanceOf[Integer])
    ).toMap
    sc.broadcast(broadcast_data)
  }
}