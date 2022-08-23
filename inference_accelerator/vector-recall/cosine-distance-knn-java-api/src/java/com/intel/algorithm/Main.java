package com.intel.algorithm;

import org.apache.commons.validator.GenericValidator;

/**
 * For test use.
 */
public class Main {

  public static void main(String[] args) {
    int res;
    if (GenericValidator.matchRegexp(args[0], ".*\\.csv") && GenericValidator.matchRegexp(args[1], ".*\\.csv")) {
      res = CosineDistanceKNN.search(6, args[0], args[1]);
    } else {
      res = -1;
    }
    
    if (res == -1) {
      System.out.println("Failed in calling native function.");
      return;
    }
    System.out.println("Indices Table:\n");
    output(CosineDistanceKNN.getIndicesTable());
    System.out.println("\nDistances Table:\n");
    output(CosineDistanceKNN.getDistancesTable());
  }

  public static <T> void output(Table<T> table) {
    int rowCount = table.getRowCount();
    int columnCount = table.getColumnCount();
    T[] data = table.getTableData();
    for (int i = 0; i < rowCount; i++) {
      for (int j = 0; j < columnCount; j++) {
        int index = i * columnCount + j;
        System.out.print(data[index] + " ");
      }
      System.out.println();
    }
  }
}