"""
Main file.

Creates an ImageClassifier and train it on the animal dataset.
Should be either:
 - Launched with the -i flag to interact with the classifier. (recommanded)
 - Completed on the files you want to test and then launched.

Must not be imported.
"""


if __name__ != "__main__":
  raise ImportError("This file can't be loaded as a module.")


from classifier import ImageClassifier


classifier = ImageClassifier("animals")