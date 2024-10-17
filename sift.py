if __name__ == "__main__":
  raise RuntimeError("This file is a module, it mustn't be run directly.")


import cv2 as cv


def compute(image):
  """
  Gets the descriptors computed by the SIFT.
  In the process, the image will be copied and grayscaled.

  :param image: The CV image matrix.
  :return: The descriptor matrix computed by the SIFT.
  """
  grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  sift = cv.SIFT_create()
  _, descriptors = sift.detectAndCompute(grayscale, None)
  return descriptors

