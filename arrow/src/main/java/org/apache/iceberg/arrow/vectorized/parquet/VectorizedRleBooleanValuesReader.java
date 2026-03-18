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
import java.nio.ByteBuffer;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.bytes.BytesUtils;
import org.apache.parquet.column.values.ValuesReader;
import org.apache.parquet.column.values.bitpacking.BytePacker;
import org.apache.parquet.column.values.bitpacking.Packer;
import org.apache.parquet.io.ParquetDecodingException;

/**
 * A {@link VectorizedValuesReader} for RLE-encoded boolean data pages.
 *
 * @see <a
 *     href="https://parquet.apache.org/docs/file-format/data-pages/encodings/#run-length-encoding--bit-packing-hybrid-rle--3">
 *     Parquet format encodings: RLE</a>
 */
public class VectorizedRleBooleanValuesReader extends ValuesReader
    implements VectorizedValuesReader {

  private enum Mode {
    RLE,
    PACKED
  }

  private static final int BIT_WIDTH = 1;
  private static final BytePacker PACKER = Packer.LITTLE_ENDIAN.newBytePacker(BIT_WIDTH);

  private ByteBufferInputStream inputStream;
  private Mode mode;
  private int currentCount;
  private int currentValue;
  private int[] packedValuesBuffer = new int[16];
  private int packedValuesBufferIdx = 0;

  @Override
  public void initFromPage(int valueCount, ByteBufferInputStream in) throws IOException {
    int length = BytesUtils.readIntLittleEndian(in);
    this.inputStream = in.sliceStream(length);
    this.currentCount = 0;
  }

  @Override
  public boolean readBoolean() {
    return nextValue() != 0;
  }

  @Override
  public void readBooleans(int total, FieldVector vec, int rowId) {
    BitVector bitVector = (BitVector) vec;
    for (int i = 0; i < total; i++) {
      bitVector.setSafe(rowId + i, readBoolean() ? 1 : 0);
    }
  }

  private int nextValue() {
    if (currentCount == 0) {
      readNextGroup();
    }

    currentCount--;
    switch (mode) {
      case RLE:
        return currentValue;
      case PACKED:
        return packedValuesBuffer[packedValuesBufferIdx++];
    }
    throw new RuntimeException("Unrecognized mode: " + mode);
  }

  private void readNextGroup() {
    try {
      int header = readUnsignedVarInt();
      this.mode = (header & 1) == 0 ? Mode.RLE : Mode.PACKED;
      switch (mode) {
        case RLE:
          this.currentCount = header >>> 1;
          this.currentValue = inputStream.read();
          return;
        case PACKED:
          int numGroups = header >>> 1;
          this.currentCount = numGroups * 8;
          if (this.packedValuesBuffer.length < this.currentCount) {
            this.packedValuesBuffer = new int[this.currentCount];
          }
          packedValuesBufferIdx = 0;
          int valueIndex = 0;
          while (valueIndex < this.currentCount) {
            ByteBuffer buffer = inputStream.slice(BIT_WIDTH);
            PACKER.unpack8Values(buffer, buffer.position(), this.packedValuesBuffer, valueIndex);
            valueIndex += 8;
          }
          return;
        default:
          throw new ParquetDecodingException("not a valid mode " + this.mode);
      }
    } catch (IOException e) {
      throw new ParquetDecodingException("Failed to read from input stream", e);
    }
  }

  private int readUnsignedVarInt() throws IOException {
    int value = 0;
    int shift = 0;
    int byteRead;
    do {
      byteRead = inputStream.read();
      value |= (byteRead & 0x7F) << shift;
      shift += 7;
    } while ((byteRead & 0x80) != 0);
    return value;
  }

  @Override
  public void skip() {
    throw new UnsupportedOperationException();
  }
}
