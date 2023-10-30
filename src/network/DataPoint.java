package network;

import java.util.Arrays;
import java.util.Objects;

public record DataPoint<T>(double[] inputs, T expectedOutput) {
  @Override
  @SuppressWarnings("unchecked")
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataPoint<T> dataPoint = (DataPoint<T>) o;
    return (
      expectedOutput == dataPoint.expectedOutput &&
      Arrays.equals(inputs, dataPoint.inputs)
    );
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(expectedOutput);
    result = 31 * result + Arrays.hashCode(inputs);
    return result;
  }

  @Override
  public String toString() {
    return (
      "DataPoint{" +
      "inputs=" +
      Arrays.toString(inputs) +
      ", expectedOutput=" +
      expectedOutput +
      '}'
    );
  }
}
