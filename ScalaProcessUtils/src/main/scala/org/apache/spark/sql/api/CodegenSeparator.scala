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

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.api.java.{UDF0, UDF1}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal, ScalaUDF}
import org.apache.spark.sql.expressions.{SparkUserDefinedFunction}
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, StringType}

class CodegenFallbackUDF(
    function: AnyRef,
    dataType: DataType,
    children: Seq[Expression],
    inputEncoders: Seq[Option[ExpressionEncoder[_]]] = Nil,
    // outputEncoder: Option[ExpressionEncoder[_]] = None,
    udfName: Option[String] = None,
    nullable: Boolean = true,
    udfDeterministic: Boolean = true
  ) extends ScalaUDF(function, dataType, children, Nil, udfName = udfName) with CodegenFallback {
    override def doGenCode(
      ctx: CodegenContext,
      ev: ExprCode): ExprCode = {
        throw new NotImplementedError("CodeGen is not supported in CodegenFallbackUDF")
      }
    /* override protected def otherCopyArgs: Seq[AnyRef] = {
      outputEncoder :: Nil
    } */
  }

case class DummyUDF() extends UDF0[String] {
  def call() = ""
}

case class DummyUDFForString() extends UDF1[String, String] {
  def call(x: String) = x
}

case class DummyUDFForInteger() extends UDF1[Int, Int] {
  def call(x: Int) = x
}

case class DummyUDFForLong() extends UDF1[Long, Long] {
  def call(x: Long) = x
}

class CodegenSeparator(
    name: String,
    f: AnyRef,
    dataType: DataType
    ) extends SparkUserDefinedFunction(f, dataType, Nil) {
  def this() {
    this("CodegenSeparator", () => (DummyUDF()).asInstanceOf[UDF0[Any]].call(), StringType)
  }
  override def createScalaUDF(exprs: Seq[Expression]): ScalaUDF = new CodegenFallbackUDF(f, dataType, exprs, udfName = Some(name))
}

class CodegenSeparator0(
    name: String,
    f: AnyRef,
    dataType: DataType
    ) extends SparkUserDefinedFunction(f, dataType, Nil) {
  def this() {
    this("CodegenSeparator", (DummyUDFForInteger()).asInstanceOf[UDF1[Int, Int]].call(_: Int), IntegerType)
  }
  override def createScalaUDF(exprs: Seq[Expression]): ScalaUDF = new CodegenFallbackUDF(f, dataType, exprs, udfName = Some(name))
}

class CodegenSeparator1(
    name: String,
    f: AnyRef,
    dataType: DataType
    ) extends SparkUserDefinedFunction(f, dataType, Nil) {
  def this() {
    this("CodegenSeparator", (DummyUDFForString()).asInstanceOf[UDF1[String, String]].call(_: String), StringType)
  }
  override def createScalaUDF(exprs: Seq[Expression]): ScalaUDF = new CodegenFallbackUDF(f, dataType, exprs, udfName = Some(name))
}

class CodegenSeparator2(
    name: String,
    f: AnyRef,
    dataType: DataType
    ) extends SparkUserDefinedFunction(f, dataType, Nil) {
  def this() {
    this("CodegenSeparator", (DummyUDFForLong()).asInstanceOf[UDF1[Long, Long]].call(_: Long), LongType)
  }
  override def createScalaUDF(exprs: Seq[Expression]): ScalaUDF = new CodegenFallbackUDF(f, dataType, exprs, udfName = Some(name))
}

