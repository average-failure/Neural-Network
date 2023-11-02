package network.cost;

public interface ICost {
  double calculateCost(double[] outputs, double[] expectedOutputs);

  double derivative(double output, double expectedOutput);
}
