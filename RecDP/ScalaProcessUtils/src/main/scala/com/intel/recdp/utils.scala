package com.intel.recdp

import java.util.UUID.randomUUID
import scala.collection.JavaConverters._

object Utils {
  def UUID(): String = {
    randomUUID().toString()
  }
  def convertToScalaArray(x: Any): Array[String] = {
    x.asInstanceOf[java.util.ArrayList[String]].asScala.toArray
  }
}
