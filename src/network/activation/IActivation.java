package network.activation;

public interface IActivation {
  double function(double weightedInput);

  double derivative(double weightedInput);
}
