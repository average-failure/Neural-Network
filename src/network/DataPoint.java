package network;

import java.util.Arrays;
import java.util.Objects;

public record DataPoint(
  double[] inputs,
  double[] expectedOutputs,
  byte expectedOutput
) {
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataPoint dataPoint = (DataPoint) o;
    return (
      expectedOutput == dataPoint.expectedOutput &&
      Arrays.equals(inputs, dataPoint.inputs) &&
      Arrays.equals(expectedOutputs, dataPoint.expectedOutputs)
    );
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(expectedOutput);
    result = 31 * result + Arrays.hashCode(expectedOutputs);
    result = 31 * result + Arrays.hashCode(inputs);
    return result;
  }

  @Override
  public String toString() {
    return String.format(
      "DataPoint [inputs=%s, expectedOutputs=%s, expectedOutput=%s",
      Arrays.toString(inputs),
      Arrays.toString(expectedOutputs),
      expectedOutput
    );
  }
}
