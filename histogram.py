if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


import numpy as np


def compute(predicts):
  """
  Compute the normalized wordbook histrogram for given descriptors.

  :param predicts: The wordbook clusters occurences.
  :return: A histogram of all occurences.
  """
  return np.histogram(predicts, bins=500, density=True)[0]