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
import scala.reflect.{ClassTag, ClassManifest}
import scala.reflect.runtime.universe.TypeTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import com.intel.recdp._

object CategorifyForMultiLevelArray {
  def process[T : ClassTag](sc: JavaSparkContext, df: DataFrame, col_name: String, dict_df: DataFrame, j_sep_list: Any): DataFrame = {
    val broadcast_data = dict_df.select("dict_col", "dict_col_id").collect().map( row => 
      if (row(0) != null) {
        (row(0).toString, row(1).asInstanceOf[Int])
      } else {
        (null, row(1).asInstanceOf[Int])
      }
    ).toMap
    val broadcast_handler = sc.broadcast(broadcast_data)
    val sep_list = Utils.convertToScalaArray(j_sep_list)
    val categorifyUDF = udf((x_l: WrappedArray[String]) => {
      val broadcasted = broadcast_handler.value
      def get_mapped(x_l: String, sep_id: Int): String = {
        if (sep_id >= sep_list.size || !x_l.contains(sep_list(sep_id))) {
          //System.out.println(s"sep is ${sep_id} and ${sep_list.toList} and x_l is ${x_l}")
          if (x_l != null && broadcasted.contains(x_l)) broadcasted(x_l).toString
          else "0"
        } else {
          x_l.split(sep_list(sep_id).toCharArray()(0)).map(x => get_mapped(x, sep_id + 1)).mkString(sep_list(sep_id))
        }
      }

      if (broadcasted == null || broadcasted.isEmpty || x_l == null) {
        Array[String]()
      } else {
        x_l.toArray.map(x => get_mapped(x, 0))
      }
    })
    return df.withColumn(col_name, categorifyUDF(col(col_name)))
  }

  def categorify(sc: JavaSparkContext, df: DataFrame, col_name: String, dict_df: DataFrame, j_sep_list: Any): DataFrame = {
    dict_df.dtypes.foreach {
      case (dict_col_name, col_type) => if (dict_col_name == "dict_col") {
        if (col_type == "StringType") {
          return process[String](sc, df, col_name, dict_df, j_sep_list)
        } else if (col_type == "IntegerType") {
          return process[Int](sc, df, col_name, dict_df, j_sep_list)
        } else if (col_type == "FloatType") {
          return process[Float](sc, df, col_name, dict_df, j_sep_list)
        } else {
          throw new NotImplementedError(s"${col_type} is currently not supported")
        }
      }
    }
    df
  }
}
