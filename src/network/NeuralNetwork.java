package network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NeuralNetwork<T> {

  private static final Random random = new Random();
  private static final ExecutorService pool = Executors.newFixedThreadPool(16);
  private static final List<Future<?>> futures = new ArrayList<>();

  private final Layer<T>[] layers;
  private final double initialLearnRate;
  private final double learnRateDecay;
  private double learnRate;
  private final double regularisation;
  private final double momentum;

  /**
   * A very basic implementation of a neural network
   * @param numLayers the number of layers in the network
   * @param numNodesPerLayer an array containing the number of nodes for each layer
   * @param outputs an array containing the value (output) of each node in the output layer
   */
  @SuppressWarnings("unchecked") // Suppress compiler warning for layer array
  public NeuralNetwork(
    int numLayers,
    int[] numNodesPerLayer,
    T[] outputs,
    double initialLearnRate,
    double learnRateDecay,
    double regularisation,
    double momentum
  ) {
    // Catch illegal parameters (not matching numbers of parameters)
    if (numNodesPerLayer.length != numLayers) {
      throw new IllegalArgumentException(
        "Number of nodes per layer must be given"
      );
    } else if (
      numNodesPerLayer[numNodesPerLayer.length - 1] != outputs.length
    ) {
      throw new IllegalArgumentException(
        "Number of outputs provided must be equal to number of nodes in output layer"
      );
    }

    // Create and populate the layers
    layers = new Layer[numLayers];

    // The input layer has no previous layer
    layers[0] = new Layer<>(numNodesPerLayer[0]);

    // Populate the hidden layers
    final int lastLayerIndex = numLayers - 1;
    for (int i = 1; i < lastLayerIndex; i++) {
      layers[i] = new Layer<>(numNodesPerLayer[i], layers[i - 1]);
    }

    // Give the last (output) layer outputs
    layers[lastLayerIndex] =
      new Layer<>(
        numNodesPerLayer[lastLayerIndex],
        layers[lastLayerIndex - 1],
        outputs
      );

    this.initialLearnRate = initialLearnRate;
    this.learnRateDecay = learnRateDecay;
    learnRate = initialLearnRate;
    this.regularisation = regularisation;
    this.momentum = momentum;
  }

  /**
   * Feeds an array of inputs to the input layer of the network
   * and returns whether the network output the correct result.
   * @param inputs the input activations to feed to the input layer
   * @param expectedOutput the expected value of the node with the highest
   * activation value in the output layer (i.e. the expected output of the network)
   * @return {@code true} if the network output the correct result, {@code false} otherwise
   */
  private boolean forwardPass(double[] inputs, T expectedOutput) {
    forwardPass(inputs);

    return getOutput() == expectedOutput;
  }

  /**
   * Feeds an array of inputs to the input layer of the network
   * @param inputs the input activations to feed to the input layer
   */
  private void forwardPass(double[] inputs) {
    // Set activations of the first (input) layer to inputs
    layers[0].setInputs(inputs);

    // Compute activations for each layer
    for (int i = 1; i < layers.length; i++) layers[i].forwardPass();
  }

  /**
   * Back-propagation
   * @param expectedOutput the expected output
   */
  private void backwardPass(T expectedOutput) {
    final Layer<T> outputLayer = layers[layers.length - 1];
    outputLayer.calculateOutputNodeValues(expectedOutput);
    outputLayer.updateGradient();

    Layer<T> hiddenLayer;
    for (int i = layers.length - 2; i > 0; i--) {
      hiddenLayer = layers[i];
      hiddenLayer.calculateNodeValues(layers[i + 1]);
      hiddenLayer.updateGradient();
    }
  }

  private T getOutput() {
    return layers[layers.length - 1].getMaxNodeValue();
  }

  public void train(DataPoint<T>[] data, int batchSize, int epochs) {
    final int[] results = new int[10];

    final int len = data.length;
    int batchIndex = 0;
    int epoch = 0;
    while (epoch < epochs) {
      trainBatch(data, batchSize, results);
      batchIndex++;
      if (batchIndex * batchSize >= len) {
        batchIndex = 0;
        epoch++;
        learnRate = (1 / (1 + learnRateDecay * epoch)) * initialLearnRate;
      }
    }

    System.out.println("\nTraining Results");
    for (int i = 0; i < results.length; i++) {
      System.out.println(i + ": " + results[i]);
    }
  }

  public void trainBatch(DataPoint<T>[] data, int batchSize, int[] results) {
    final int len = data.length;

    for (int i = 0; i < batchSize; i++) {
      futures.add(
        pool.submit(() -> {
          final DataPoint<T> dataPoint = data[random.nextInt(len)];
          forwardPass(dataPoint.inputs());
          results[(byte) getOutput()]++;
          backwardPass(dataPoint.expectedOutput());
        })
      );
    }

    futures.forEach(future -> {
      try {
        future.get();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        e.printStackTrace();
      } catch (ExecutionException e) {
        e.printStackTrace();
      }
    });
    futures.clear();

    final double divLearnRate = learnRate / len;
    for (Layer<T> layer : layers) {
      layer.applyGradient(divLearnRate, regularisation, momentum);
    }
  }

  public void test(DataPoint<T>[] data) {
    final int[] results = new int[10];

    for (DataPoint<T> dataPoint : data) {
      forwardPass(dataPoint.inputs(), dataPoint.expectedOutput());
      results[(byte) getOutput()]++;
    }

    System.out.println("\nTesting Results");
    for (int i = 0; i < results.length; i++) {
      System.out.println(i + ": " + results[i]);
    }

    pool.shutdown();
  }

  @Override
  public String toString() {
    return String.format("NeuralNetwork [layers=%s]", Arrays.toString(layers));
  }
}
