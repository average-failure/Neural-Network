package network.activation;

public class Tanh implements IActivation {

  @Override
  public void function(double[] weightedInputs) {
    for (int i = 0; i < weightedInputs.length; i++) {
      weightedInputs[i] = Math.tanh(weightedInputs[i]);
    }
  }

  @Override
  public double[] derivative(double[] weightedInputs) {
    final double[] derivatives = new double[weightedInputs.length];
    for (int i = 0; i < derivatives.length; i++) {
      final double sech = 1 / Math.cosh(weightedInputs[i]);
      derivatives[i] = sech * sech;
    }
    return derivatives;
  }
}
