import pandas as pd
from Naive_Bayes import Naive_Bayes

COLUMN_NAMES = [
  "class",
  "x-box",
  "y-box",
  "width",
  "high",
  "onpix",
  "x-bar",
  "y-bar",
  "x2bar",
  "y2bar",
  "xybar",
  "x2ybr",
  "xy2br",
  "x-ege",
  "xegvy",
  "y-ege",
  "yegvx",
]

RANDOM_STATE = 42
TRAIN_SIZE = 16000


def main():
  df = pd.read_csv("dataset/letter-recognition.data", header=None, names=COLUMN_NAMES)
  df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
  train_data = df.iloc[:TRAIN_SIZE].reset_index(drop=True)
  test_data = df.iloc[TRAIN_SIZE:].reset_index(drop=True)

  nb = Naive_Bayes(train_data, label_col="class")
  nb.test(test_data)


if __name__ == "__main__":
  main()

