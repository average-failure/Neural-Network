package network.activation;

import java.util.Arrays;

public class Sigmoid implements IActivation {

  /**
   * Math sigmoid function to squish values between 0 and 1.
   * @param weightedInputs the weighted inputs to feed into the function
   */
  @Override
  public void function(double[] weightedInputs) {
    for (int i = 0; i < weightedInputs.length; i++) {
      weightedInputs[i] = 1 / (1 + exp(-weightedInputs[i]));
    }
  }

  /**
   * The derivative of the sigmoid function.
   * @param weightedInputs the weighted inputs to feed into the function
   * @return an array containing the weighted inputs after the function
   */
  @Override
  public double[] derivative(double[] weightedInputs) {
    final double[] derivatives = Arrays.copyOf(
      weightedInputs,
      weightedInputs.length
    );
    function(derivatives);
    for (int i = 0; i < derivatives.length; i++) {
      derivatives[i] = derivatives[i] * (1 - derivatives[i]);
    }
    return derivatives;
  }

  private static double exp(double val) {
    final long tmp = (long) (1512775 * val + 1072632447);
    return Double.longBitsToDouble(tmp << 32);
  }
}
