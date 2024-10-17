if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


from sklearn.cluster import KMeans
import pickle


def compute(descriptors, n_clusters=500):
  """
  Compute the clusters which represents the visual words.

  :param descriptors: The descriptor concatenated matrix.
  :param n_clusters:  The number of visual words wanted.
  :return: The fitted KMeans object.
  """
  clusters = KMeans(n_clusters=n_clusters)
  clusters.fit(descriptors)
  return clusters


def save(model):
  """
  Saves model inside the kmeans.sav.

  :param model: The kmeans model to save.
  """
  pickle.dump(model, open("kmeans.sav", "wb"))


def load():
  """
  Loads model from the kmeans.sav.

  :return: The saved kmeans model.
  """
  return pickle.load(open("kmeans.sav", "rb"))