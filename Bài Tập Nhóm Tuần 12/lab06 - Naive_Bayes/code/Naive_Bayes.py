import numpy as np


class Naive_Bayes:
  def __init__(self, data_set, label_col=-1):
    self.ds = data_set.copy()
    if isinstance(label_col, str):
      self.label_col = label_col
    else:
      self.label_col = self.ds.columns[label_col]
    self.feature_cols = [col for col in self.ds.columns if col != self.label_col]
    grouped = self.ds.groupby(self.label_col)
    self.ds_means = grouped[self.feature_cols].mean()
    self.ds_variances = grouped[self.feature_cols].var().replace(0, 1e-6)
    self.class_probabilities = self.get_class_probabilities(self.ds)

  def get_class_probabilities(self, data_set):
    class_sizes = data_set.groupby(self.label_col).size()
    ds_total = data_set.shape[0]
    probs = {}
    for cls, size in class_sizes.items():
      probs[cls] = size / ds_total
    return probs

  def get_probability_density(self, x, mean, variance):
    variance = max(variance, 1e-6)
    pd = 1 / (np.sqrt(2 * np.pi * variance)) * np.exp((-(x - mean)**2) / (2 * variance))
    return pd

  def predict(self, x):
    feature_class_probabilities = {}
    for group, class_prob in self.class_probabilities.items():
      prob = class_prob
      for value, col in zip(x, self.feature_cols):
        prob *= self.get_probability_density(value, self.ds_means.loc[group][col], self.ds_variances.loc[group][col])
      feature_class_probabilities[group] = prob
    feature_class = max(feature_class_probabilities, key=feature_class_probabilities.get)
    return feature_class

  def test(self, test_data):
    correct = 0
    total = 0
    for _, row in test_data.iterrows():
      feature_set = row[self.feature_cols].values
      group = self.predict(feature_set)
      if group == row[self.label_col]:
        correct += 1
      else:
        print(tuple(feature_set), "prediction=", group, "correct=", row[self.label_col])
      total += 1
    accuracy = correct / total if total else 0
    print("Accuracy=", accuracy)