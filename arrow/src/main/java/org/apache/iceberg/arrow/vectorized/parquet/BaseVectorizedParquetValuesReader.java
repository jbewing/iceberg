/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.arrow.vectorized.parquet;

import java.io.IOException;
import org.apache.parquet.bytes.BytesUtils;

/**
 * A values reader for Parquet's run-length encoded data that reads column data in batches instead
 * of one value at a time. This is based off of the VectorizedRleValuesReader class in Apache Spark
 * with these changes:
 *
 * <p>Writes batches of values retrieved to Arrow vectors. If all pages of a column within the row
 * group are not dictionary encoded, then dictionary ids are eagerly decoded into actual values
 * before writing them to the Arrow vectors
 */
@SuppressWarnings("checkstyle:VisibilityModifier")
public class BaseVectorizedParquetValuesReader
    extends VectorizedRLEAndBitPackingParquetValuesReader {

  private int bytesWidth;

  // If true, the bit width is fixed. This decoder is used in different places and this also
  // controls if we need to read the bitwidth from the beginning of the data stream.
  private final boolean fixedWidth;
  private final boolean readLength;
  final int maxDefLevel;

  final boolean setArrowValidityVector;

  public BaseVectorizedParquetValuesReader(int maxDefLevel, boolean setValidityVector) {
    this.maxDefLevel = maxDefLevel;
    this.fixedWidth = false;
    this.readLength = false;
    this.setArrowValidityVector = setValidityVector;
  }

  public BaseVectorizedParquetValuesReader(
      int bitWidth, int maxDefLevel, boolean setValidityVector) {
    this(bitWidth, maxDefLevel, bitWidth != 0, setValidityVector);
  }

  public BaseVectorizedParquetValuesReader(
      int bitWidth, int maxDefLevel, boolean readLength, boolean setValidityVector) {
    this.fixedWidth = true;
    this.readLength = readLength;
    this.maxDefLevel = maxDefLevel;
    this.setArrowValidityVector = setValidityVector;
    init(bitWidth);
  }

  @Override
  protected void init(int bw) {
    super.init(bw);
    this.bytesWidth = BytesUtils.paddedByteCountFromBits(bw);
  }

  @Override
  int encodedDataLength() throws IOException {
    if (fixedWidth) {
      return readLength ? readIntLittleEndian() : -1;
    } else {
      if (inputStream.available() > 0) {
        init(inputStream.read());
      }
      return -1;
    }
  }

  /** Reads the next byteWidth little endian int. */
  @Override
  int readIntLittleEndianPaddedOnBitWidth() throws IOException {
    switch (bytesWidth) {
      case 0:
        return 0;
      case 1:
        return inputStream.read();
      case 2:
        {
          int ch2 = inputStream.read();
          int ch1 = inputStream.read();
          return (ch1 << 8) + ch2;
        }
      case 3:
        {
          int ch3 = inputStream.read();
          int ch2 = inputStream.read();
          int ch1 = inputStream.read();
          return (ch1 << 16) + (ch2 << 8) + ch3;
        }
      case 4:
        {
          return readIntLittleEndian();
        }
    }
    throw new RuntimeException("Non-supported bytesWidth: " + bytesWidth);
  }
}
