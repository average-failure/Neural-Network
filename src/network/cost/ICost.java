package network.cost;

public interface ICost {
  double calculateCost(double[] outputs, double[] expectedOutputs);

  double function(double output, double expectedOutput);

  double derivative(double output, double expectedOutput);
}
