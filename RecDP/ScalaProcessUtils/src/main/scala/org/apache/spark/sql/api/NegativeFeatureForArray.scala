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
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.reflect.{ClassTag, ClassManifest}
import scala.reflect.runtime.universe.TypeTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType}

// This operator aims to add negative sample as new row
object NegativeFeatureForArray {
  def process[T : ClassTag](sc: JavaSparkContext, df: DataFrame, tgt_name: String, col_name: String, dict_df: DataFrame, neg_cnt: Int): DataFrame = {
    val broadcast_data = dict_df.select("dict_col").collect().map( row => 
      row(0).asInstanceOf[T]
    )
    val broadcasted = sc.broadcast(broadcast_data)
    val addNegativeFeatureUDF = if (neg_cnt == 1) {
      udf((hist_asin: WrappedArray[Any]) => {
        val r = new Random
        val item_list = broadcasted.value
        val num_items = item_list.size
        if (hist_asin == null) {
          Array[String]()
        } else {
          hist_asin.toArray.map(asin_any => {
            val asin = asin_any.asInstanceOf[T]
            var asin_neg = asin
            do {
                val asin_neg_index: Int = r.nextInt(num_items - 1)
                asin_neg = item_list(asin_neg_index)
            } while (asin_neg == asin)
            asin_neg.toString
          })
        }
      })
    } else {
      udf((hist_asin: WrappedArray[Any]) => {
        val r = new Random
        val item_list = broadcasted.value
        val num_items = item_list.size
        if (hist_asin == null) {
          Array[String]()
        } else {
          hist_asin.toArray.map(asin_any => {
            val asin = asin_any.asInstanceOf[T]
            var smp_cnt = 0
            var res = ArrayBuffer[String]()
            do {
              var asin_neg = asin
              do {
                  val asin_neg_index: Int = r.nextInt(num_items - 1)
                  asin_neg = item_list(asin_neg_index)
              } while (asin_neg == asin)
              res += asin_neg.toString
              smp_cnt += 1
            } while (smp_cnt < neg_cnt)
            res.mkString("|")
          })
        }
      })
    }
    return df.withColumn(tgt_name, addNegativeFeatureUDF(col(col_name)))
  }
  def add(sc: JavaSparkContext, df: DataFrame, tgt_name: String, col_name: String, dict_df: DataFrame, neg_cnt: Int): DataFrame = {
    dict_df.dtypes.foreach {
      case (dict_col_name, col_type) => if (dict_col_name == "dict_col") {
        if (col_type == "StringType") {
          return process[String](sc, df, tgt_name, col_name, dict_df, neg_cnt)
        } else if (col_type == "IntegerType") {
          return process[Int](sc, df, tgt_name, col_name, dict_df, neg_cnt)
        } else if (col_type == "FloatType") {
          return process[Float](sc, df, tgt_name, col_name, dict_df, neg_cnt)
        } else {
          throw new NotImplementedError(s"${col_type} is currently not supported")
        }
      }
    }
    df
  }
}
