package network.trainer;

import java.util.Arrays;
import java.util.Random;
import network.DataPoint;
import network.core.NeuralNetwork;

public class NetworkTrainer<T> {

  private static final Random RANDOM = new Random();

  private final NeuralNetwork<T> network;

  private final Batch<DataPoint>[] batches;
  private final double initialLearnRate;
  private double currentLearnRate;
  private final double learnRateDecay;
  private int batchIndex;
  private int epochCount;

  /**
   * @param params the training parameters
   * @param trainingData the training data
   */
  public NetworkTrainer(TrainerParams params, DataPoint[] trainingData) {
    this.batches = splitData(trainingData, params.miniBatchSize());

    network =
      new NeuralNetwork<>(
        params.layerSizes(),
        params.regularisation(),
        params.momentum()
      );

    initialLearnRate = params.initialLearnRate();
    currentLearnRate = initialLearnRate;
    learnRateDecay = params.learnRateDecay();
  }

  public void run(int iterations) {
    for (int i = 0; i < iterations; i++) {
      network.learn(batches[batchIndex].data, currentLearnRate);
      batchIndex++;

      if (batchIndex >= batches.length) epochCompleted();
    }
    System.out.println("Current learn rate: " + currentLearnRate);
  }

  private void epochCompleted() {
    batchIndex = 0;
    epochCount++;
    shuffleArray(batches);
    currentLearnRate =
      (1 / (1 + learnRateDecay * epochCount)) * initialLearnRate;
  }

  public int[] test(DataPoint[] testingData) {
    shuffleArray(testingData);

    final int[] results = new int[testingData[0].expectedOutputs().length];

    for (DataPoint testData : testingData) {
      results[getMaxIndex(network.calculateOutputs(testData.inputs()))]++;
    }

    return results;
  }

  public double testAccuracy(DataPoint[] testingData) {
    shuffleArray(testingData);

    int correct = 0;

    for (DataPoint testData : testingData) {
      if (
        testData.expectedOutput() ==
        getMaxIndex(network.calculateOutputs(testData.inputs()))
      ) correct++;
    }

    return (double) correct / testingData.length;
  }

  private static int getMaxIndex(double[] values) {
    int maxIndex = 0;

    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[maxIndex]) maxIndex = i;
    }

    return maxIndex;
  }

  @SuppressWarnings("unchecked")
  private static <D> Batch<D>[] splitData(D[] data, int batchSize) {
    shuffleArray(data);

    final Batch<D>[] batches = new Batch[data.length / batchSize];
    for (int i = 0; i < batches.length; i++) {
      batches[i] =
        new Batch<>(
          Arrays.copyOfRange(data, i * batchSize, (i + 1) * batchSize)
        );
    }

    return batches;
  }

  private static void shuffleArray(Object[] array) {
    int elementsRemaining = array.length;
    int randomIndex;

    while (elementsRemaining > 1) {
      randomIndex = RANDOM.nextInt(elementsRemaining);
      final Object chosenElement = array[randomIndex];

      elementsRemaining--;
      array[randomIndex] = array[elementsRemaining];
      array[elementsRemaining] = chosenElement;
    }
  }

  private static class Batch<D> {

    D[] data;

    Batch(D[] data) {
      this.data = data;
    }
  }
}
