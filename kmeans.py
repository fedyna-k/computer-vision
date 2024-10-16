if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


from sklearn.cluster import KMeans


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