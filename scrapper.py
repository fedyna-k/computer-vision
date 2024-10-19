if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


from sklearn.model_selection import train_test_split
import cv2 as cv
import os


def get_all_pictures(path: str, verbose=True):
  """
  Get all images and their corresponding labels.
  The labels are automatically computed from parent folders.

  :param path: The base path of all images.
  :param verbose: Is every step printed?
  :return: Train test images and train test labels.
  """
  subpaths = list(map(lambda p: path + "/" + p, os.listdir(path)))
  labels_dic = []
  labels = []
  images = []

  while subpaths:
    path = subpaths.pop()
    if os.path.isdir(path):
      subpaths += list(map(lambda p: path + "/" + p, os.listdir(path)))
    else:
      dirpath = "/".join(path.split("/")[:-1])
      if dirpath not in labels_dic:
        labels_dic.append(dirpath)

      labels.append(labels_dic.index(dirpath))
      images.append(cv.imread(path))
  
  verbose and print("All labels:", labels_dic)

  return *train_test_split(images, labels, test_size=0.2), labels_dic