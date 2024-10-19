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
from matplotlib.pyplot import hist, show
from numpy import mean


knn = []
svc = []

for i in range(50):
  print(f"{100 * i // 49}%")
  classifier = ImageClassifier("animals", verbose=False)
  score = classifier.get_scores()

  knn.append(score["knn"])
  svc.append(score["svc"])

print("Average KNN:", mean(knn))
print("Average SVC:", mean(svc))

hist([knn, svc], histtype="bar")
show()