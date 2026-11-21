"""Classify one image with the shipped PCA and SVC.

    python classify.py path/to/image.jpg

Prints cloudy, rain, shine or sunrise. The feature extraction is the pipeline of
notebooks/FinalProject.ipynb, the same code the README shows.
"""
import sys

import cv2 as cv
import joblib
import numpy as np


def znorm(v):
    v = np.asarray(v, dtype="float32")
    std = v.std()
    return (v - v.mean()) / std if std != 0 else v


def histograms(img, hsv=False):
    if hsv:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    planes = cv.split(img)
    return [znorm(cv.calcHist(planes, [i], None, [256], (0, 256))) for i in range(3)]


def features(img):
    bgr = histograms(img)                                    # 3 x 256
    contrast = znorm(cv.meanStdDev(img))                     # 6
    small = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (100, 100),
                      interpolation=cv.INTER_NEAREST)
    pixels = np.float32(small.reshape(-1, 3))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, centers = cv.kmeans(pixels, 6, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = znorm(centers[np.argsort(counts)[::-1]])      # 6 x 3
    dog = cv.GaussianBlur(img, (1, 1), 0) - cv.GaussianBlur(img, (3, 3), 0)
    dog_hsv = histograms(dog, hsv=True)                      # 3 x 256

    out = np.float32((0, 0))                                 # padding, as in the notebook
    for part in bgr + [contrast, dominant] + dog_hsv:
        out = np.concatenate((out, np.array(part, dtype="float32").flatten()))
    return out                                               # (1562,)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python classify.py path/to/image.jpg")
    image = cv.imread(sys.argv[1])
    if image is None:
        sys.exit(f"cannot read {sys.argv[1]}")
    pca = joblib.load("models/pca_weatherClassification.pkl")
    svc = joblib.load("models/svc_weatherClassification.pkl")
    print(svc.predict(pca.transform([features(image)]))[0])
