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

import scala.util.Try

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.api.java.{UDF0, UDF1}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal, ScalaUDF}
import org.apache.spark.sql.expressions.{SparkUserDefinedFunction}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType, LongType, StringType}

class CategorifyUDF(broadcast_handler: Broadcast[Map[String, Integer]]) extends UDF1[String, Integer] {
  def call(x: String): Integer = {   
    val broadcasted = broadcast_handler.value
    if (broadcasted == null || broadcasted.isEmpty || x == null) return -1.asInstanceOf[Integer]
    if (broadcasted.contains(x)) broadcasted(x) else -1.asInstanceOf[Integer]
  }
}

class Categorify(
    name: String,
    f: AnyRef,
    dataType: DataType
    ) extends SparkUserDefinedFunction(f, dataType, Nil, None, Some(name), true, true) {
      def this(broadcasted: Broadcast[Map[String, Integer]]) = {
        this("Categorify", (new CategorifyUDF(broadcasted)).asInstanceOf[UDF1[Any, Any]].call(_: Any), IntegerType)
      }
}