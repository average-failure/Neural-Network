package network.cost;

public class MeanSquaredError implements ICost {

  /**
   * Calculates the cost of the network through the mean squared error function.
   * @param outputs the outputs of the network
   * @param expectedOutputs the expected outputs of the network
   * @return the cost of the network
   */
  @Override
  public double calculateCost(double[] outputs, double[] expectedOutputs) {
    double cost = 0;
    for (int i = 0; i < outputs.length; i++) {
      cost += function(outputs[i], expectedOutputs[i]);
    }
    return cost;
  }

  /**
   * The mean squared error function
   * @param output the output of the node
   * @param expectedOutput the expected output of the node
   * @return the mean squared error function at the point {@code output}
   */
  @Override
  public double function(double output, double expectedOutput) {
    final double error = output - expectedOutput;
    return error * error;
  }

  /**
   * The derivative of the mean squared error function
   * @param output the output of the node
   * @param expectedOutput the expected output of the node
   * @return the derivative of {@link MeanSquaredError#function} at the point {@code output}
   */
  @Override
  public double derivative(double output, double expectedOutput) {
    return 2 * (output - expectedOutput);
  }
}
