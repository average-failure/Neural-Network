package network.core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import network.DataPoint;

public class NeuralNetwork<T> {

  private static final ExecutorService pool = Executors.newFixedThreadPool(16);
  private static final List<Future<?>> futures = new ArrayList<>();

  private final Layer<T>[] layers;
  private LearnData[] batchLearnData;
  private final double regularisation;
  private final double momentum;

  /**
   * A very basic implementation of a neural network
   * @param layerSizes an array containing the number of nodes for each layer
   * @param regularisation the regularisation of the network
   * @param momentum the momentum of the network
   */
  @SuppressWarnings("unchecked") // Suppress compiler warning for layer array
  public NeuralNetwork(
    int[] layerSizes,
    double regularisation,
    double momentum
  ) {
    // Create and populate the layers
    layers = new Layer[layerSizes.length - 1];

    // Populate the hidden layers
    for (int i = 0; i < layers.length; i++) {
      layers[i] = new Layer<>(layerSizes[i], layerSizes[i + 1]);
    }

    this.regularisation = regularisation;
    this.momentum = momentum;
  }

  /**
   * Feeds an array of inputs to the input layer of the network
   * @param inputs the input activations to feed to the input layer
   * @return the outputs of the network
   */
  public double[] calculateOutputs(double[] inputs) {
    for (Layer<T> layer : layers) inputs = layer.forwardPass(inputs);
    return inputs;
  }

  /**
   * Feeds the {@code dataPoint} through the network then uses back-propagation
   * to compute the gradient of the cost function at that {@code dataPoint};
   * @param dataPoint the data point to feed to the network
   * @param learnData the learn data of this network
   */
  private void updateGradients(DataPoint dataPoint, LearnData learnData) {
    // Feed data through network to calculate outputs
    double[] inputsToNextLayer = dataPoint.inputs();

    for (int i = 0; i < layers.length; i++) {
      inputsToNextLayer =
        layers[i].forwardPass(inputsToNextLayer, learnData.layerData[i]);
    }

    System.out.println(
      Layer.COST.calculateCost(inputsToNextLayer, dataPoint.expectedOutputs())
    );

    // Back-propagation
    final int outputIndex = layers.length - 1;
    final Layer<T> outputLayer = layers[outputIndex];
    final Layer.LearnData outputData = learnData.layerData[outputIndex];

    // Update output layer gradients
    outputLayer.calculateOutputNodeValues(
      outputData,
      dataPoint.expectedOutputs()
    );
    outputLayer.updateGradients(outputData);

    // Update all hidden layer gradients
    for (int i = outputIndex - 1; i >= 0; i--) {
      final Layer.LearnData layerLearnData = learnData.layerData[i];

      layers[i].calculateNodeValues(
          layerLearnData,
          layers[i + 1],
          learnData.layerData[i + 1].nodeValues
        );
      layers[i].updateGradients(layerLearnData);
    }
  }

  public void learn(DataPoint[] data, double learnRate) {
    if (batchLearnData == null || batchLearnData.length != data.length) {
      batchLearnData = new LearnData[data.length];
      for (int i = 0; i < batchLearnData.length; i++) {
        batchLearnData[i] = new LearnData(layers);
      }
    }

    for (int i = 0; i < data.length; i++) {
      final int iCopy = i;
      submitTask(() -> updateGradients(data[iCopy], batchLearnData[iCopy]));
    }
    blockThread();

    // Apply the gradients to the weights and biases of each layer
    for (Layer<T> layer : layers) {
      layer.applyGradients(learnRate / data.length, regularisation, momentum);
    }
  }

  /**
   * A helper method for multithreading that submits
   * a task to the executor thread pool.
   * @param task the task to submit to the pool
   */
  private static void submitTask(Runnable task) {
    futures.add(pool.submit(task));
  }

  /**
   * A helper method for multithreading that blocks the current thread
   * until all tasks in the {@link #futures} array have been completed.
   * Clears the array after.
   */
  private static void blockThread() {
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
  }

  public static void shutdown() {
    pool.shutdown();
  }

  public static class LearnData {

    final Layer.LearnData[] layerData;

    /**
     * @param layers the array of layers to store in the learn data
     */
    public LearnData(Layer<?>[] layers) {
      layerData = new Layer.LearnData[layers.length];
      for (int i = 0; i < layers.length; i++) {
        layerData[i] = new Layer.LearnData(layers[i]);
      }
    }
  }
}
