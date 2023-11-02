package network.trainer;

import java.util.Arrays;
import java.util.Objects;

public record TrainerParams(
  int[] layerSizes,
  double initialLearnRate,
  double learnRateDecay,
  double regularisation,
  double momentum,
  int miniBatchSize
) {
  public TrainerParams(int[] layerSizes) {
    this(layerSizes, 0.6, 0.075, 0.1, 0.9, 32);
    System.out.println("Initial learn rate: " + initialLearnRate);
    System.out.println("Learn rate decay: " + learnRateDecay);
    System.out.println("Regularisation: " + regularisation);
    System.out.println("Momentum: " + momentum);
    System.out.println("Mini batch size: " + miniBatchSize);
    System.out.println();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    TrainerParams trainerParams = (TrainerParams) o;
    return (
      initialLearnRate == trainerParams.initialLearnRate &&
      learnRateDecay == trainerParams.learnRateDecay &&
      regularisation == trainerParams.regularisation &&
      momentum == trainerParams.momentum &&
      miniBatchSize == trainerParams.miniBatchSize &&
      Arrays.equals(layerSizes, trainerParams.layerSizes)
    );
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(initialLearnRate);
    result = 31 * result + Objects.hash(initialLearnRate);
    result = 31 * result + Objects.hash(learnRateDecay);
    result = 31 * result + Objects.hash(regularisation);
    result = 31 * result + Objects.hash(momentum);
    result = 31 * result + Objects.hash(miniBatchSize);
    result = 31 * result + Arrays.hashCode(layerSizes);
    return result;
  }

  @Override
  public String toString() {
    return String.format(
      "TrainerParams [layerSizes=%s, initialLearnRate=%s, learnRateDecay=%s, regularisation=%s, momentum=%s, miniBatchSize=%s]",
      Arrays.toString(layerSizes),
      initialLearnRate,
      learnRateDecay,
      regularisation,
      momentum,
      miniBatchSize
    );
  }
}
