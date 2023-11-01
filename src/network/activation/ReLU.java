package network.activation;

public class ReLU implements IActivation {

  /**
   * Rectified Linear Unit function
   * @param weightedInput the weighted input to feed into the function
   * @return {@code weightedInput} if it is greater than 0, {@code 0} otherwise
   */
  @Override
  public double function(double weightedInput) {
    return weightedInput > 0 ? weightedInput : 0;
  }

  /**
   * Derivative of the ReLU function
   * @param weightedInput the weighted input to feed into the function
   * @return {@code 1} if {@code weightedInput} is greater than 0, {@code 0} otherwise
   */
  @Override
  public double derivative(double weightedInput) {
    return weightedInput > 0 ? 1 : 0;
  }
}
