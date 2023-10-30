package network;

import java.util.Arrays;

public class Layer<T> {

  private final Node<T>[] nodesIn;
  public final Node<T>[] nodes;

  /**
   * The constructor for the hidden layers in a neural network
   * @param numNodes the number of nodes in this layer
   * @param previousLayer the layer before (closer to the input layer) this layer
   */
  @SuppressWarnings("unchecked") // Suppress compiler warning for node array
  public Layer(int numNodes, Layer<T> previousLayer) {
    nodes = new Node[numNodes];
    nodesIn = previousLayer.nodes;

    final int len = nodesIn.length;
    for (int i = 0; i < numNodes; i++) nodes[i] = new Node<>(len);
  }

  /**
   * The constructor for the output layer in a neural network
   * @param numNodes the number of nodes in this layer
   * @param previousLayer the layer before (closer to the input layer) this layer
   * @param values the output values of this layer
   */
  @SuppressWarnings("unchecked") // Suppress compiler warning for node array
  public Layer(int numNodes, Layer<T> previousLayer, T[] values) {
    if (values.length != numNodes) throw new IllegalArgumentException(
      "A value must be provided for each node"
    );

    nodes = new Node[numNodes];
    nodesIn = previousLayer.nodes;

    final int len = nodesIn.length;
    for (int i = 0; i < numNodes; i++) nodes[i] = new Node<>(values[i], len);
  }

  /**
   * The constructor for the input layer in a neural network
   * @param numNodes the number of nodes in this layer
   */
  @SuppressWarnings("unchecked") // Suppress compiler warning for node array
  public Layer(int numNodes) {
    nodes = new Node[numNodes];
    nodesIn = null;

    for (int i = 0; i < numNodes; i++) nodes[i] = new Node<>();
  }

  /**
   * The method to compute the activations of the nodes
   * in this layer from the previous layer's nodes.
   */
  public void forwardPass() {
    for (Node<T> node : nodes) {
      double weightedInput = node.getBias();
      final double[] weights = node.getWeights();
      for (int i = 0; i < nodesIn.length; i++) {
        weightedInput += nodesIn[i].getActivation() * weights[i];
      }
      node.setWeightedInput(weightedInput);
      node.setActivation(activationFunction(weightedInput));
    }
  }

  /**
   * Calculates the cost gradient for this layer.
   */
  public void updateGradient() {
    for (Node<T> node : nodes) node.updateGradient(nodesIn);
  }

  /**
   * Calculates the node values for each node.
   * <p>This method is only meant for the output layer.</p>
   * @param expectedOutput the expected output of the network
   */
  public void calculateOutputNodeValues(T expectedOutput) {
    for (Node<T> node : nodes) node.calculateNodeValue(expectedOutput);
  }

  /**
   * Calculates the node values of this layer.
   */
  public void calculateNodeValues(Layer<T> oldLayer) {
    for (int nodeOut = 0; nodeOut < nodes.length; nodeOut++) {
      double nodeValue = 0;
      for (int oldNode = 0; oldNode < oldLayer.nodes.length; oldNode++) {
        final double weightedInputDerivative =
          oldLayer.nodes[oldNode].getWeights()[nodeOut];
        nodeValue +=
          weightedInputDerivative * oldLayer.nodes[oldNode].getNodeValue();
      }
      nodeValue *= activationDerivative(nodes[nodeOut].getWeightedInput());
      nodes[nodeOut].setNodeValue(nodeValue);
    }
  }

  /**
   * Finds the node with the highest activation in this layer and returns its value.
   * <p>This method is only meant for the output layer.</p>
   * @return the value of the node with the highest activation in this layer
   */
  public T getMaxNodeValue() {
    Node<T> max = nodes[0];
    for (int i = 1; i < nodes.length; i++) {
      if (nodes[i].getActivation() > max.getActivation()) max = nodes[i];
    }
    return max.getValue();
  }

  /**
   * Sets the activations of each node in this layer.
   * <p>This method is only meant for the input layer.</p>
   * @param inputs the input activations to set for each node in this layer
   */
  public void setInputs(double[] inputs) {
    if (nodes.length != inputs.length) throw new IllegalArgumentException(
      "Number of inputs must be equal to number of nodes"
    );

    for (int i = 0; i < nodes.length; i++) {
      nodes[i].setActivation(inputs[i]);
    }
  }

  /**
   * Apply each node's gradient to its weights and biases
   */
  public void applyGradient(
    double learnRate,
    double regularisation,
    double momentum
  ) {
    final double weightDecay = 1 - regularisation * learnRate;
    for (int i = 0; i < nodes.length; i++) {
      nodes[i].applyGradient(weightDecay, learnRate, momentum);
    }
  }

  /**
   * Calculates the cost of the network based on the activations of the output layer.
   * The cost is calculated by {@code sum(square(activation - expected output))}.
   * <p>This method is only meant for the output layer.</p>
   * @param expectedOutput the expected output of the network
   * @return the cost of the network
   */
  public double calculateCost(T expectedOutput) {
    double cost = 0;
    for (Node<T> node : nodes) {
      final boolean isExpectedOutput = node.getValue() == expectedOutput;
      cost += costFunction(node.getActivation(), isExpectedOutput);
    }
    return cost;
  }

  /**
   * The cost function as defined in {@link Layer#calculateCost} javadoc
   * @param activation the activation of the node
   * @param isExpectedOutput whether the node is the expected output of the network or not
   * @return the cost function at the point {@code activation}
   */
  private static double costFunction(
    double activation,
    boolean isExpectedOutput
  ) {
    final double error = activation - (isExpectedOutput ? 1 : 0);
    return error * error;
  }

  /**
   * The derivative of the cost function
   * @param activation the activation of the node
   * @param isExpectedOutput whether the node is the expected output of the network or not
   * @return the derivative of {@link Layer#costFunction} at the point {@code activation}
   */
  public static double costDerivative(
    double activation,
    boolean isExpectedOutput
  ) {
    return 2 * (activation - (isExpectedOutput ? 1 : 0));
  }

  /**
   * Math sigmoid function to squish values between 0 and 1.
   * This method should only be called by {@link Layer#activationFunction}.
   * @param x any real number
   * @return a double between 0 and 1
   */
  private static double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * The derivative of the sigmoid function.
   * This method should only be called by {@link Layer#activationDerivative}.
   * @param x any real number
   * @return the derivative of the sigmoid function at the point {@code x}
   */
  private static double sigmoidDerivative(double x) {
    final double y = sigmoid(x);
    return y * (1 - y);
  }

  /**
   * This method is used for easy change to the activation function.
   * @param x any real number
   * @return the number after it is passed through the activation function
   */
  private static double activationFunction(double x) {
    return sigmoid(x);
  }

  /**
   * This method is used for easy change to the activation function.
   * @param x any real number
   * @return the number after it is passed through the derivative of the activation function
   */
  public static double activationDerivative(double x) {
    return sigmoidDerivative(x);
  }

  /**
   * Rectified Linear Unit function to prevent the need for {@link Math#exp}. Same use as {@link Layer#sigmoid}.
   * @param x any real number
   * @return {@code x} if it is greater than 0, {@code 0} otherwise
   */
  private static double reLU(double x) {
    return x > 0 ? x : 0;
  }

  /**
   * Derivative of the ReLU function
   * @param x any real number
   * @return {@code 1} if x is greater than 0, {@code 0} otherwise
   */
  private static double reLUDerivative(double x) {
    return x > 0 ? 1 : 0;
  }

  @Override
  public String toString() {
    return String.format("Layer [nodes=%s]", Arrays.toString(nodes));
  }
}
