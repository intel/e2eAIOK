package com.intel.algorithm;

/**
 * Benchmark test.
 */
public class BenchmarkTest {

  public static void main(String[] args) {
    int rows_train_count = args.length < 1 ? 1000000 : Integer.valueOf(args[0]);
    int columns_count = args.length < 2 ? 512 : Integer.valueOf(args[1]);
    int rows_query_count = args.length < 3 ? 100 : Integer.valueOf(args[2]);
    int neighbors_count = args.length < 4 ? 10 : Integer.valueOf(args[3]);
    System.out.println("rows_train_count: " + rows_train_count + ", columns_count: "
      + columns_count + ", rows_query_count: " + rows_query_count + ", neighbors_count: "
      + neighbors_count);
    int res = CosineDistanceKNN.benchmark(rows_train_count, columns_count,
      rows_query_count, neighbors_count);
    if (res == -1) {
      System.out.println("Failed in calling native function.");
      return;
    }
  }
}
