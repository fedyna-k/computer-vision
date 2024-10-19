if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


import numpy as np
import cv2 as cv
import scrapper
import sift
import kmeans
import histogram

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class ImageClassifier:
  """
  Image classifier.
  """


  def __init__(self, path, fast=True, verbose=True):
    """
    Instanciates a new ImageClassifier object.
    Note that the training time might take some time.
    Will recursively go through all images.

    :param path: The path of the training images folder.
    :param fast: Is the pretrained model loaded?
    :param verbose: Is every step printed?
    """
    verbose and print("Reading images...")
    train, test, label_train, label_test, self.__LABELS = scrapper.get_all_pictures(path, verbose=verbose)

    verbose and print("Computing descriptors...")
    descriptors_train = list(map(sift.compute, train))
    descriptors_test = list(map(sift.compute, test))

    verbose and print("Concatenating matrices...")
    descriptors_mat_train = np.concatenate(descriptors_train)
    descriptors_mat_test = np.concatenate(descriptors_test)

    verbose and print("Fitting KMeans...")
    if fast:
      self.__KMEANS = kmeans.load()
    else:
      self.__KMEANS = kmeans.compute(descriptors_mat_train, n_clusters=100)
      kmeans.save(self.__KMEANS)

    verbose and print("Extracting closest clusters...")
    predicts_train = list(map(self.__KMEANS.predict, descriptors_train))
    predicts_test = list(map(self.__KMEANS.predict, descriptors_test))

    verbose and print("Creating histograms for features...")
    features_train = np.array(list(map(histogram.compute, predicts_train)))
    features_test = np.array(list(map(histogram.compute, predicts_test)))

    verbose and print("Creating KNN...")

    self.__KNN = self.__optimize(
      lambda k: KNeighborsClassifier(k),
      range(1, 31),
      lambda k: k,
      features_train, features_test, label_train, label_test
    )

    self.__KNN.fit(features_train, label_train)
    self.__KNN_SCORE = self.__KNN.score(features_test, label_test)
    verbose and print(f"KNN score after training: {self.__KNN_SCORE}")

    verbose and print("Creating SVC...")

    self.__SVC = self.__optimize(
      lambda C: SVC(C=C),
      range(-3, 4),
      lambda i: pow(10, i),
      features_train, features_test, label_train, label_test
    )
    
    self.__SVC.fit(features_train, label_train)
    self.__SVC_SCORE = self.__SVC.score(features_test, label_test)
    verbose and print(f"SVC score after training: {self.__SVC_SCORE}")

  
  def __optimize(self, classifier_creator, parameter_range, parameter_computation, features_train, features_test, label_train, label_test):
    """
    Optimize the given model on one hyperparameter.

    :param classifier_creator: The function that creates the model.
    :param parameter_range: The hyperparameter computation function input range.
    :param parameter_computation: The hyperparameter computation function.
    :param features_train: The train set of features.
    :param features_test: The test set of features.
    :param label_train: The train set of labels.
    :param label_test: The test set of labels.
    :return: The optimized model on the given hyperparameter for these sets.
    """
    score_max = 0
    parameter_max = 0

    for i in parameter_range:
      classifier = classifier_creator(parameter_computation(i))
      classifier.fit(features_train, label_train)
      score = classifier.score(features_test, label_test)
      if score > score_max:
        score_max = score
        parameter_max = parameter_computation(i)
    
    return classifier_creator(parameter_max)


  def __get_features(self, image):
    """
    Gets all features for a given image.

    :param image: The image cv matrix.
    :return: The histogram for the image.
    """
    descriptors = sift.compute(image)
    predicts = self.__KMEANS.predict(descriptors)
    return histogram.compute(predicts)

 
  def predict(self, path):
    """
    Predicts the label for a given image.

    :param self: The class instance.
    :param path: The image path.
    :return: The predicted labels for the image.
    """
    image = cv.imread(path)
    features = self.__get_features(image)
    return {
      "knn": self.__LABELS[self.__KNN.predict(features.reshape(1, -1))[0]],
      "svc": self.__LABELS[self.__SVC.predict(features.reshape(1, -1))[0]]
    }
  
  def get_scores(self):
    """
    Gets the KNN and SVC models scores.

    :return: The score
    """
    return {
      "knn": self.__KNN_SCORE,
      "svc": self.__SVC_SCORE
    }