package network;

import java.util.Arrays;

public class Node<T> {

  private final double[] weights;
  private final double[] costGradientWeight;
  private final double[] weightVelocities;
  private double costGradientBias;
  private double biasVelocity;

  private double weightedInput;
  private double activation;
  private double bias;
  private double nodeValue;
  private final T value;

  /**
   * The constructor for nodes in the output layer of a neural network
   * @param value the value of this node
   * @param numNodesIn the number of nodes in the previous layer of the network
   */
  public Node(T value, int numNodesIn) {
    this.value = value;
    weights = new double[numNodesIn];
    costGradientWeight = new double[numNodesIn];
    weightVelocities = new double[numNodesIn];
    Arrays.setAll(weights, i -> Math.random());
    activation = Math.random();
    bias = Math.random();
  }

  /**
   * The constructor for nodes in hidden layers of a neural network
   * @param numNodesIn the number of nodes in the previous layer of the network
   */
  public Node(int numNodesIn) {
    this(null, numNodesIn);
  }

  /**
   * The constructor for nodes in the input layer of a neural network
   */
  public Node() {
    this(null, 0);
  }

  /**
   * @return the weights of the connections to the previous layer's nodes
   */
  public double[] getWeights() {
    return weights;
  }

  /**
   * @return the activation value of this node
   */
  public double getActivation() {
    return activation;
  }

  /**
   * @param activation the activation to set
   */
  public void setActivation(double activation) {
    this.activation = activation;
  }

  /**
   * @return the bias value of this node
   */
  public double getBias() {
    return bias;
  }

  /**
   * @return the output value of this node
   */
  public T getValue() {
    return value;
  }

  /**
   * @return the weightedInput
   */
  public double getWeightedInput() {
    return weightedInput;
  }

  /**
   * @param weightedInput the weightedInput to set
   */
  public void setWeightedInput(double weightedInput) {
    this.weightedInput = weightedInput;
  }

  /**
   * @return the nodeValue
   */
  public double getNodeValue() {
    return nodeValue;
  }

  /**
   * This method is only meant to be called from nodes in the output layer of a neural network.
   * @param expectedOutput the expected output for the network
   */
  public void calculateNodeValue(T expectedOutput) {
    nodeValue =
      Layer.activationDerivative(weightedInput) *
      Layer.costDerivative(activation, value.equals(expectedOutput));
  }

  /**
   * @param nodeValue the nodeValue to set
   */
  public void setNodeValue(double nodeValue) {
    this.nodeValue = nodeValue;
  }

  /**
   * Apply the gradients to the weights and bias
   * @param weightDecay the weight decay of the network
   * @param learnRate the learning rate of the network
   * @param momentum the momentum of the network
   */
  public void applyGradient(
    double weightDecay,
    double learnRate,
    double momentum
  ) {
    // Apply weight gradients
    for (int i = 0; i < weights.length; i++) {
      weightVelocities[i] =
        weightVelocities[i] * momentum - costGradientWeight[i] * learnRate;
      weights[i] = weights[i] * weightDecay + weightVelocities[i];
      costGradientWeight[i] = 0;
    }

    // Apply bias gradient
    biasVelocity = biasVelocity * momentum - costGradientBias * learnRate;
    bias += biasVelocity;
    costGradientBias = 0;
  }

  /**
   * Updates the weight and bias gradients for this node
   * @param nodesIn the nodes in the previous layer of the network (i.e. the input nodes to this layer)
   */
  public synchronized void updateGradient(Node<T>[] nodesIn) {
    // Update weight gradients
    for (int nodeIn = 0; nodeIn < nodesIn.length; nodeIn++) {
      final double costWeightDerivative =
        nodesIn[nodeIn].activation * nodeValue;
      costGradientWeight[nodeIn] += costWeightDerivative;
    }

    // Update bias gradient
    final double costBiasDerivative = nodeValue;
    costGradientBias += costBiasDerivative;
  }

  @Override
  public String toString() {
    return String.format(
      "Node [weights=%s, weightedInput=%s, activation=%s, bias=%s, nodeValue=%s, value=%s]",
      Arrays.toString(weights),
      weightedInput,
      activation,
      bias,
      nodeValue,
      value
    );
  }
}
