# USAGE
# python build_dataset.py --dataset /raid/datasets/flowers17/flowers17

# import the necessary packages
from imutils import paths
import argparse
import random
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path dataset of input images")
args = vars(ap.parse_args())

# grab all image paths and create a training and testing split
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.shuffle(imagePaths)
i = int(len(imagePaths) * 0.75)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# define the datasets
datasets = [
	("training", trainPaths, "flowers17_training.csv"),
	("testing", testPaths, "flowers17_testing.csv")
]

# loop over the data splits
for (dType, imagePaths, outputPath) in datasets:
	# open the output CSV file for writing
	print("[INFO] building '{}' split...".format(dType))
	f = open(outputPath, "w")

	# loop over all input images
	for imagePath in imagePaths:
		# load the input image and resize it to 64x64 pixels
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (64, 64))

		# create a flattened list of pixel values
		image = [str(x) for x in image.flatten()]

		# extract the class label from the file path and write the
		# label along pixels list to disk
		label = imagePath.split(os.path.sep)[-2]
		f.write("{},{}\n".format(label, ",".join(image)))

	# close the output CSV file
	f.close()