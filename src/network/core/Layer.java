package network.core;

import java.util.Arrays;
import network.activation.IActivation;
import network.activation.Sigmoid;
import network.cost.ICost;
import network.cost.MeanSquaredError;

public class Layer<T> {

  public static final IActivation ACTIVATION = new Sigmoid();
  public static final ICost COST = new MeanSquaredError();

  private final int numNodesIn;
  private final int numNodesOut;

  private final double[] weights;
  private final double[] biases;

  private final double[] costGradientWeight;
  private final double[] costGradientBias;

  private final double[] weightVelocities;
  private final double[] biasVelocities;

  /**
   * The constructor for a layer in a neural network
   * @param numNodesIn the number of nodes in the previous layer of the network
   * @param numNodesOut the number of nodes in this layer
   */
  public Layer(int numNodesIn, int numNodesOut) {
    this.numNodesIn = numNodesIn;
    this.numNodesOut = numNodesOut;

    weights = new double[numNodesIn * numNodesOut];
    biases = new double[numNodesOut];

    costGradientWeight = new double[weights.length];
    costGradientBias = new double[biases.length];

    weightVelocities = new double[weights.length];
    biasVelocities = new double[biases.length];

    Arrays.parallelSetAll(weights, i -> Math.random() * 2 - 1);
    Arrays.setAll(biases, i -> Math.random() * 2 - 1);
  }

  private double getWeight(int nodeIn, int nodeOut) {
    return weights[getWeightIndex(nodeIn, nodeOut)];
  }

  private int getWeightIndex(int nodeIn, int nodeOut) {
    return nodeOut * numNodesIn + nodeIn;
  }

  /**
   * Computes the activations of the nodes
   * in this layer from the previous layer's nodes.
   * @param inputs the inputs from the previous layer
   * @return the output activations from this layer
   */
  public double[] forwardPass(double[] inputs) {
    // Calculate activations
    final double[] activations = new double[numNodesOut];
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      double weightedInput = biases[nodeOut];
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        weightedInput += inputs[nodeIn] * getWeight(nodeIn, nodeOut);
      }
      activations[nodeOut] = ACTIVATION.function(weightedInput);
    }

    return activations;
  }

  /**
   * Calculates the activations of this layer and stores it in
   * the given {@link LearnData}.
   * @param inputs the inputs from the previous layer
   * @param learnData the learn data of this layer
   * @return the output activations from this layer
   */
  public double[] forwardPass(double[] inputs, LearnData learnData) {
    learnData.inputs = inputs;

    // Calculate weighted inputs and activations
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      double weightedInput = biases[nodeOut];
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        weightedInput += inputs[nodeIn] * getWeight(nodeIn, nodeOut);
      }
      learnData.weightedInputs[nodeOut] = weightedInput;
      learnData.activations[nodeOut] = ACTIVATION.function(weightedInput);
    }

    return learnData.activations;
  }

  /**
   * Apply this layer's gradients to each weight and bias.
   * @param learnRate the learning rate
   * @param regularisation the regularisation
   * @param momentum the momentum
   */
  public void applyGradients(
    double learnRate,
    double regularisation,
    double momentum
  ) {
    final double weightDecay = 1 - regularisation * learnRate;

    for (int i = 0; i < weights.length; i++) {
      final double weight = weights[i];
      final double velocity =
        weightVelocities[i] * momentum - costGradientWeight[i] * learnRate;
      weightVelocities[i] = velocity;
      weights[i] = weight * weightDecay + velocity;
      costGradientWeight[i] = 0;
    }

    for (int i = 0; i < biases.length; i++) {
      final double velocity =
        biasVelocities[i] * momentum - costGradientBias[i] * learnRate;
      biasVelocities[i] = velocity;
      biases[i] += velocity;
      costGradientBias[i] = 0;
    }
  }

  /**
   * Calculates the node values for each node in the output layer using the
   * partial derivative of the cost with respect to the the weighted input.
   * <p>This method is only meant for the output layer.</p>
   * @param learnData the learn data of this layer
   * @param expectedOutputs the expected outputs of the network
   */
  public void calculateOutputNodeValues(
    LearnData learnData,
    double[] expectedOutputs
  ) {
    for (int i = 0; i < learnData.nodeValues.length; i++) {
      final double costDerivative = COST.derivative(
        learnData.activations[i],
        expectedOutputs[i]
      );
      final double activationDerivative = ACTIVATION.derivative(
        learnData.weightedInputs[i]
      );
      learnData.nodeValues[i] = costDerivative * activationDerivative;
    }
  }

  /**
   * Calculates the node values for each node in the output layer using the
   * partial derivative of the cost with respect to the the weighted input.
   * @param learnData the learn data of this layer
   * @param oldLayer the layer after this layer in the network
   * @param oldNodeValues the node values of the {@code oldLayer}
   */
  public void calculateNodeValues(
    LearnData learnData,
    Layer<T> oldLayer,
    double[] oldNodeValues
  ) {
    for (int newNode = 0; newNode < numNodesOut; newNode++) {
      double newNodeValue = 0;
      for (int oldNode = 0; oldNode < oldNodeValues.length; oldNode++) {
        final double weightedInputDerivative = oldLayer.getWeight(
          newNode,
          oldNode
        );
        newNodeValue += weightedInputDerivative * oldNodeValues[oldNode];
      }
      newNodeValue *= ACTIVATION.derivative(learnData.weightedInputs[newNode]);
      learnData.nodeValues[newNode] = newNodeValue;
    }
  }

  /**
   * Calculates the cost gradients for this layer.
   * @param learnData the learn data for this layer
   */
  public void updateGradients(LearnData learnData) {
    synchronized (costGradientWeight) {
      for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        final double nodeValue = learnData.nodeValues[nodeOut];
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
          final double costWeightDerivative =
            learnData.inputs[nodeIn] * nodeValue;
          costGradientWeight[getWeightIndex(nodeIn, nodeOut)] +=
            costWeightDerivative;
        }
      }

      synchronized (costGradientBias) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
          final double costBiasDerivative = learnData.nodeValues[nodeOut];
          costGradientBias[nodeOut] += costBiasDerivative;
        }
      }
    }
  }

  public static class LearnData {

    double[] inputs;
    final double[] weightedInputs;
    final double[] activations;
    final double[] nodeValues;

    /**
     * @param layer the layer to create the learn data for
     */
    public LearnData(Layer<?> layer) {
      weightedInputs = new double[layer.numNodesOut];
      activations = new double[layer.numNodesOut];
      nodeValues = new double[layer.numNodesOut];
    }
  }
}
