package com.intel.algorithm;

/**
 * Maintains indices/distances table metadata obtained from cosine distance based
 * KNN algorithm.
 * The whole table data is kept off-heap. With metadata here, we can get
 * specific data or specific rows of data.
 */
public class TableWithOnlyMetadata {

  private long rowCount;
  private long columnCount;
  // table address on native memory.
  private long addr;
}
