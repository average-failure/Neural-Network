package network.cost;

public class CrossEntropyLoss implements ICost {

  @Override
  public double calculateCost(double[] outputs, double[] expectedOutputs) {
    double cost = 0;

    for (int i = 0; i < outputs.length; i++) {
      final double x =
        expectedOutputs[i] *
        Math.log(outputs[i]) +
        (1 - expectedOutputs[i]) *
        Math.log(1 - outputs[i]);
      if (!Double.isNaN(x)) cost += x;
    }

    return -cost / outputs.length;
  }

  @Override
  public double derivative(double output, double expectedOutput) {
    return (output - expectedOutput) / (output * (1 - output));
  }
}
