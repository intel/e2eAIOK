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

package com.intel.recdp

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}
import scala.collection.mutable.WrappedArray

class SortIntArrayByFrequency extends UDF1[WrappedArray[Integer], Array[Integer]] {
  def call(x: WrappedArray[Integer]) = x.toArray.groupBy(identity).toArray.sortBy(-_._2.size).map(_._1)
}

class SortStringArrayByFrequency extends UDF1[WrappedArray[String], Array[String]] {
  def call(x: WrappedArray[String]) = x.toArray.groupBy(identity).toArray.sortBy(-_._2.size).map(_._1)
}
