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
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.bytes.BytesUtils;
import org.apache.parquet.column.values.ValuesReader;

/**
 * A {@link VectorizedValuesReader} implementation for the encoding type Run Length Encoding / RLE.
 *
 * @see <a
 *     href="https://parquet.apache.org/docs/file-format/data-pages/encodings/#run-length-encoding--bit-packing-hybrid-rle--3">
 *     Parquet format encodings: RLE</a>
 */
public class VectorizedRleBooleanValuesReader extends ValuesReader
    implements VectorizedValuesReader {

  // Since we can only read booleans, bit-width is always 1
  private static final int BOOLEAN_BIT_WIDTH = 1;
  // Since this can only be used in the context of a data page, the definition level can be set to
  // anything, and it doesn't really matter
  private static final int IRRELEVANT_MAX_DEFINITION_LEVEL = 1;
  // This class reads the length prefix itself in initFromPage, so the delegate must not read it
  private static final boolean DONT_READ_LENGTH = false;

  private final BaseVectorizedParquetValuesReader delegate;

  public VectorizedRleBooleanValuesReader(boolean setArrowValidityVector) {
    this.delegate =
        new BaseVectorizedParquetValuesReader(
            BOOLEAN_BIT_WIDTH,
            IRRELEVANT_MAX_DEFINITION_LEVEL,
            DONT_READ_LENGTH,
            setArrowValidityVector);
  }

  @Override
  public void initFromPage(int valueCount, ByteBufferInputStream in) throws IOException {
    // For boolean data pages (v1 and v2), the RLE-encoded data is prefixed with a 4-byte
    // little-endian length. We read and strip this prefix here before delegating to the
    // base RLE decoder. See https://parquet.apache.org/docs/file-format/data-pages/encodings/#RLE
    int length = BytesUtils.readIntLittleEndian(in);
    ByteBufferInputStream slicedStream = in.sliceStream(length);
    delegate.initFromPage(valueCount, slicedStream);
  }

  @Override
  public boolean readBoolean() {
    return delegate.readBoolean();
  }

  @Override
  public void readBooleans(int total, FieldVector vec, int rowId) {
    BitVector bitVector = (BitVector) vec;
    for (int i = 0; i < total; i++) {
      bitVector.setSafe(rowId + i, delegate.readBoolean() ? 1 : 0);
    }
  }

  @Override
  public void skip() {
    throw new UnsupportedOperationException();
  }
}
