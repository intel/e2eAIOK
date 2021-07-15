package com.intel.algorithm;

/**
 * Maintains indices/distances obtained from the reference of cosine distance
 * KNN algorithm for given data.
 */
public class Table<T> {

  private int rowCount;
  private int columnCount;

  private T[] data;

  public Table(int rowCount, int columnCount) {
    this.rowCount = rowCount;
    this.columnCount = columnCount;
    this.data = (T[]) new Object[rowCount * columnCount];
  }

  public void setData(int ind, T d) {
    data[ind] = d;
  }

  public int getRowCount() {
    return rowCount;
  }

  public int getColumnCount() {
    return columnCount;
  }

  public T[] getTableData() {
    return data;
  }
}
