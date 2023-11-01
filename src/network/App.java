package network;

import java.io.DataInputStream;
import java.io.IOException;
import network.core.NeuralNetwork;
import network.trainer.NetworkTrainer;
import network.trainer.TrainerParams;

public class App<T> {

  private static final int IMAGE_SIZE = 28 * 28;

  public static void main(String[] args) throws IOException {
    new App<Byte>().start(new int[] { IMAGE_SIZE, 16, 16, 10 });
  }

  private void start(int[] layerSizes) throws IOException {
    final TrainerParams params = new TrainerParams(
      layerSizes,
      0.05,
      0.075,
      0.1,
      0.9,
      32
    );
    final DataPoint[] trainingInputs = readData(
      "assets/train-images.idx3-ubyte",
      "assets/train-labels.idx1-ubyte"
    );
    final NetworkTrainer<T> trainer = new NetworkTrainer<>(
      params,
      trainingInputs
    );

    long startTime = System.currentTimeMillis();
    trainer.run(3000);
    long endTime = System.currentTimeMillis();
    System.out.println("Training time: " + (endTime - startTime) + "ms");

    final DataPoint[] testingInputs = readData(
      "assets/t10k-images.idx3-ubyte",
      "assets/t10k-labels.idx1-ubyte"
    );

    startTime = System.currentTimeMillis();
    // final int[] results = trainer.test(testingInputs);
    final double accuracy = trainer.testAccuracy(testingInputs);
    endTime = System.currentTimeMillis();
    System.out.println("Testing time: " + (endTime - startTime) + "ms");
    // System.out.println("Testing Results: ");
    // for (int i = 0; i < results.length; i++) {
    //   System.out.println(i + ": " + results[i]);
    // }
    System.out.println("Testing accuracy: " + (accuracy * 100) + "%");

    NeuralNetwork.shutdown();
  }

  private static DataPoint[] readData(String imagesPath, String labelsPath)
    throws IOException {
    final ClassLoader classLoader = App.class.getClassLoader();
    final DataInputStream imageStream = new DataInputStream(
      classLoader.getResourceAsStream(imagesPath)
    );
    final DataInputStream labelStream = new DataInputStream(
      classLoader.getResourceAsStream(labelsPath)
    );

    imageStream.readInt();
    final int numImages = imageStream.readInt();

    if (IMAGE_SIZE != imageStream.readInt() * imageStream.readInt()) {
      throw new IllegalStateException("Image size is not correct");
    }

    labelStream.readInt();
    labelStream.readInt();

    final DataPoint[] dataPoints = new DataPoint[numImages];
    final byte[] imageData = new byte[IMAGE_SIZE];

    for (int i = 0; i < numImages; i++) {
      imageStream.readFully(imageData);
      final double[] image = new double[IMAGE_SIZE];
      for (int j = 0; j < IMAGE_SIZE; j++) {
        image[j] = (imageData[j] & 0xFF) / 255d;
      }

      final byte label = labelStream.readByte();
      final double[] expectedOutputs = new double[10];
      expectedOutputs[label] = 1;
      dataPoints[i] = new DataPoint(image, expectedOutputs, label);
    }

    imageStream.close();
    labelStream.close();

    return dataPoints;
  }
}
