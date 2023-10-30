package network;

import java.io.DataInputStream;
import java.io.IOException;

public class App {

  private static final int IMAGE_SIZE = 28 * 28;

  public static void main(String[] args) throws IOException {
    final NeuralNetwork<Byte> network = new NeuralNetwork<>(
      4,
      new int[] { IMAGE_SIZE, 16, 16, 10 },
      new Byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
      0.05,
      0.075,
      0.1,
      0.9
    );

    final DataPoint<Byte>[] trainingInputs = readData(
      "assets/train-images.idx3-ubyte",
      "assets/train-labels.idx1-ubyte"
    );
    network.train(trainingInputs, 32, 10);

    final DataPoint<Byte>[] testingInputs = readData(
      "assets/t10k-images.idx3-ubyte",
      "assets/t10k-labels.idx1-ubyte"
    );
    network.test(testingInputs);
  }

  @SuppressWarnings("unchecked")
  private static DataPoint<Byte>[] readData(
    String imagesPath,
    String labelsPath
  ) throws IOException {
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

    final DataPoint<Byte>[] dataPoints = new DataPoint[numImages];
    final byte[] imageData = new byte[IMAGE_SIZE];

    for (int i = 0; i < numImages; i++) {
      imageStream.readFully(imageData);
      final double[] image = new double[IMAGE_SIZE];
      for (int j = 0; j < IMAGE_SIZE; j++) {
        image[j] = (imageData[j] & 0xFF) / 255d;
      }

      final byte label = labelStream.readByte();
      dataPoints[i] = new DataPoint<>(image, label);
    }

    imageStream.close();
    labelStream.close();

    return dataPoints;
  }
}
