package com.intel.algorithm;

import org.apache.commons.validator.GenericValidator;

/**
 * Benchmark test.
 */
public class BenchmarkTest {

  public static void main(String[] args) {
    int rows_train_count, columns_count, rows_query_count, neighbors_count;
    try {
      rows_train_count = Integer.parseInt(args[0]);
      columns_count = Integer.parseInt(args[1]);
      rows_query_count = Integer.parseInt(args[2]);
      neighbors_count = Integer.parseInt(args[3]);
    } catch (NumberFormatException e) {
      rows_train_count = 1000000;
      columns_count = 512;
      rows_query_count = 100;
      neighbors_count = 10;
    }
    System.out.println("rows_train_count: " + rows_train_count + ", columns_count: "
      + columns_count + ", rows_query_count: " + rows_query_count + ", neighbors_count: "
      + neighbors_count);
    if (rows_train_count > 0 && columns_count > 0 && rows_query_count > 0 && neighbors_count > 0) {
      int res = CosineDistanceKNN.benchmark(rows_train_count, columns_count),
        rows_query_count, neighbors_count);
    } else {
      System.out.println("Failed in calling native function.");
      return;
    }
  }
}
