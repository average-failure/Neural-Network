package network.activation;

public interface IActivation {
  void function(double[] weightedInputs);

  double[] derivative(double[] weightedInputs);
}
