package network.activation;

import java.util.Arrays;

public class SoftMax implements IActivation {

  @Override
  public void function(double[] weightedInputs) {
    final double max = getMax(weightedInputs);

    double sum = 0;
    for (int i = 0; i < weightedInputs.length; i++) {
      weightedInputs[i] = Math.exp(weightedInputs[i] - max);
      sum += weightedInputs[i];
    }
    if (sum == 0) sum = 1;

    for (int i = 0; i < weightedInputs.length; i++) {
      weightedInputs[i] /= sum;
    }
  }

  @Override
  public double[] derivative(double[] weightedInputs) {
    final int len = weightedInputs.length;

    final double[] softMax = Arrays.copyOf(weightedInputs, len);
    function(softMax);

    final double[][] jacobian = new double[len][len];
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < len; j++) {
        if (i == j) {
          jacobian[i][j] = softMax[i] * (1.0 - softMax[i]);
        } else {
          jacobian[i][j] = -softMax[i] * softMax[j];
        }
      }
    }

    final double[] derivatives = new double[len];
    for (int i = 0; i < len; i++) {
      double sum = 0;
      for (int j = 0; j < len; j++) {
        sum += jacobian[i][j];
      }
      derivatives[i] = sum;
    }

    return derivatives;
  }

  private static double getMax(double[] arr) {
    double max = arr[0];
    for (int i = 1; i < arr.length; i++) {
      if (arr[i] > max) max = arr[i];
    }
    return max;
  }
}
