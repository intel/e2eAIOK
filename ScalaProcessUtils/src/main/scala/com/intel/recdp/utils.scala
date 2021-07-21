package com.intel.recdp

import java.util.UUID.randomUUID

object Utils {
  def UUID(): String = {
    randomUUID().toString()
  }
}
