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
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;

/**
 * A {@link VectorizedValuesReader} for RLE-encoded boolean data pages.
 *
 * @see <a
 *     href="https://parquet.apache.org/docs/file-format/data-pages/encodings/#run-length-encoding--bit-packing-hybrid-rle--3">
 *     Parquet format encodings: RLE</a>
 */
public class VectorizedRleBooleanValuesReader extends VectorizedRLEAndBitPackingParquetValuesReader
    implements VectorizedValuesReader {

  @Override
  int encodedDataLength() throws IOException {
    init(1);
    return readIntLittleEndian();
  }

  @Override
  int readIntLittleEndianPaddedOnBitWidth() throws IOException {
    return inputStream.read();
  }

  @Override
  public void readBooleans(int total, FieldVector vec, int rowId) {
    BitVector bitVector = (BitVector) vec;
    for (int i = 0; i < total; i++) {
      bitVector.setSafe(rowId + i, readBoolean() ? 1 : 0);
    }
  }
}
