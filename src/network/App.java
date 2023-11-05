package network;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;
import javax.swing.SwingUtilities;
import network.core.NeuralNetwork;
import network.swing.DisplayFrame;
import network.trainer.NetworkTrainer;
import network.trainer.TrainerParams;

public class App<T> {

  public static final byte IMAGE_SIDE = 28;
  public static final int IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE;
  public static final boolean DEBUG = false;
  private static final boolean TRANSFORM = false;
  private static final Random RANDOM = new Random();

  public static void main(String[] args) {
    SwingUtilities.invokeLater(DisplayFrame::new);
  }

  public NetworkTrainer<T> start(int[] layerSizes, int iterations)
    throws IOException {
    final TrainerParams params = new TrainerParams(layerSizes);
    final DataPoint[] trainingInputs = readData(
      "assets/train-images.idx3-ubyte",
      "assets/train-labels.idx1-ubyte"
    );
    final NetworkTrainer<T> trainer = new NetworkTrainer<>(
      params,
      trainingInputs
    );

    System.out.println("Starting Training...");
    System.out.println();

    long startTime = System.currentTimeMillis();
    trainer.run(iterations);
    System.out.println(
      "Training time: " + (System.currentTimeMillis() - startTime) + "ms"
    );

    final DataPoint[] testingInputs = readData(
      "assets/t10k-images.idx3-ubyte",
      "assets/t10k-labels.idx1-ubyte"
    );

    final byte NUM_TESTS = 5;
    double accuracy = 0;
    long testTime = 0;

    for (int i = 0; i < NUM_TESTS; i++) {
      startTime = System.currentTimeMillis();
      accuracy += trainer.testAccuracy(testingInputs);
      testTime += System.currentTimeMillis() - startTime;
    }

    System.out.println(
      "Average testing time: " + (testTime / NUM_TESTS) + "ms"
    );
    System.out.println(
      "Average testing accuracy: " + (accuracy / NUM_TESTS * 100) + "%"
    );

    NeuralNetwork.shutdown();

    return trainer;
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

      final Double[][] tempImage = new Double[IMAGE_SIDE][IMAGE_SIDE];
      for (int j = 0; j < IMAGE_SIDE; j++) {
        final Double[] innerImage = new Double[IMAGE_SIDE];
        for (int k = 0; k < IMAGE_SIDE; k++) {
          innerImage[k] = (imageData[j * k] & 0xFF) / 255d;
        }
        tempImage[j] = innerImage;
      }

      if (TRANSFORM) {
        final int xOff = RANDOM.nextInt(-3, 3);
        final int yOff = RANDOM.nextInt(-3, 3);
        for (int j = 0; j < tempImage.length; j++) {
          shiftArray(tempImage[j], xOff);
        }
        shiftArray(tempImage, yOff);

        rotateArray(tempImage, RANDOM.nextDouble(-0.1, 0.1));
      }

      final double[] image = flatten(tempImage)
        .mapToDouble(Double.class::cast)
        .toArray();

      final byte label = labelStream.readByte();
      final double[] expectedOutputs = new double[10];
      expectedOutputs[label] = 1;
      dataPoints[i] = new DataPoint(image, expectedOutputs, label);
    }

    imageStream.close();
    labelStream.close();

    return dataPoints;
  }

  private static void rotateArray(Double[][] matrix, double theta) {
    final int rows = matrix.length;
    final int cols = matrix[0].length;
    final double centerX = cols / 2d;
    final double centerY = rows / 2d;

    final double cosTheta = Math.cos(theta);
    final double sinTheta = Math.sin(theta);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        final double x = j - centerX;
        final double y = i - centerY;

        // Apply the rotation formula
        final double newX = x * cosTheta - y * sinTheta + centerX;
        final double newY = x * sinTheta + y * cosTheta + centerY;

        if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
          // Interpolate the rotated value
          matrix[i][j] = interpolate(matrix, newX, newY);
        } else {
          matrix[i][j] = 0d; // Set to 0 or any default value for out-of-bounds pixels
        }
      }
    }
  }

  private static double interpolate(Double[][] matrix, double x, double y) {
    final int x1 = (int) x;
    final int y1 = (int) y;
    final int x2 = x1 + 1;
    final int y2 = y1 + 1;

    if (
      !(x1 >= 0 && x2 < matrix[0].length && y1 >= 0 && y2 < matrix.length)
    ) return 0;

    final double dx = x - x1;
    final double dy = y - y1;

    return (
      matrix[y1][x1] *
      (1 - dx) *
      (1 - dy) +
      matrix[y2][x1] *
      (1 - dx) *
      dy +
      matrix[y1][x2] *
      dx *
      (1 - dy) +
      matrix[y2][x2] *
      dx *
      dy
    );
  }

  public static Stream<Object> flatten(Object[] array) {
    return Arrays
      .stream(array)
      .flatMap(o -> o instanceof Object[] ? flatten((Object[]) o) : Stream.of(o)
      );
  }

  private static void shiftArray(Object[] arr, int k) {
    if (k < 0) k += arr.length;
    k %= arr.length;

    reverse(arr, 0, arr.length - 1);
    reverse(arr, 0, k - 1);
    reverse(arr, k, arr.length - 1);
  }

  private static void reverse(Object[] arr, int start, int end) {
    while (start < end) {
      final Object temp = arr[start];
      arr[start] = arr[end];
      arr[end] = temp;
      start++;
      end--;
    }
  }
}
