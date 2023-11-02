package network.activation;

public class ReLU implements IActivation {

  /**
   * Rectified Linear Unit function
   * @param weightedInputs the weighted inputs to feed into the function
   */
  @Override
  public void function(double[] weightedInputs) {
    for (int i = 0; i < weightedInputs.length; i++) {
      if (weightedInputs[i] < 0) weightedInputs[i] = 0;
    }
  }

  /**
   * Derivative of the ReLU function
   * @param weightedInputs the weighted inputs to feed into the function
   * @return an array containing the weighted inputs after the function
   */
  @Override
  public double[] derivative(double[] weightedInputs) {
    final double[] derivatives = new double[weightedInputs.length];
    for (int i = 0; i < derivatives.length; i++) {
      if (weightedInputs[i] > 0) derivatives[i] = 1;
    }
    return derivatives;
  }
}
