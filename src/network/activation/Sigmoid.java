package network.activation;

public class Sigmoid implements IActivation {

  /**
   * Math sigmoid function to squish values between 0 and 1.
   * @param weightedInput the weighted input to feed into the function
   * @return a double between 0 and 1
   */
  @Override
  public double function(double weightedInput) {
    return 1 / (1 + Math.exp(-weightedInput));
  }

  /**
   * The derivative of the sigmoid function.
   * @param weightedInput the weighted input to feed into the function
   * @return the derivative of the sigmoid function at the point {@code weightedInput}
   */
  @Override
  public double derivative(double weightedInput) {
    final double sigmoid = function(weightedInput);
    return sigmoid * (1 - sigmoid);
  }
}
