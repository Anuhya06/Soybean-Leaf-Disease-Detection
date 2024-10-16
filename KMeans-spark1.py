from __future__ import print_function
import sys
import csv
import datetime
import time
import glob
import cv2
import os, re
import math;
from pyspark.sql import SparkSession

import random
from copy import copy, deepcopy

import numpy as np
from random import randint

import pyspark;
from pyspark import SparkContext
import sklearn.metrics.pairwise;
rbf=sklearn.metrics.pairwise.rbf_kernel;

def kmeans(data, k, max_iters=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of points assigned to each cluster
        new_centroids = []
        for i in range(k):
            if np.any(labels == i):
                new_centroids.append(data[labels == i].mean(axis=0))
            else:
                # Handle empty clusters by reassigning a random data point as the centroid
                new_centroids.append(data[np.random.choice(len(data))])
        new_centroids = np.array(new_centroids)

        # Check for convergence
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return centroids, labels

from sklearn.metrics import silhouette_score

def k_means_clustering(image_path, k_clusters):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    pixels = image.reshape(-1, 3)

    # Perform K-Means clustering on the image pixels
    cluster_centers, labels = kmeans(pixels, k_clusters)

    # Calculate the silhouette score for this image
    score = silhouette_score(pixels, labels, metric='euclidean')

    # Replace pixel values with cluster center values
    segmented_image = cluster_centers[labels].reshape(image.shape).astype(np.uint8)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return segmented_image, image_path, score

def save_segmented_image(segmented_image, image_path, output_directory):
    filename = os.path.basename(image_path)
    output_file = os.path.join(output_directory, filename)
    cv2.imwrite(output_file, segmented_image)


if __name__ == "__main__":
	parent_directory = '/home/iiti/Desktop/Soybean_Project/NobgData'
	output_directory= '/home/iiti/Desktop/Soybean_Project/NobgKmeans'
	
	sc = SparkContext(appName="KMeans1")
	k_clusters = 16
	print("started")
	# Specify the number of clusters
	K = 100  # You can change this value as needed

	start_time = datetime.datetime.now()

	# Create an RDD to hold the segmented images
	segmented_images = sc.emptyRDD()

	# Iterate through subdirectories to process each label
	for label_directory in os.listdir(parent_directory):
		label_path = os.path.join(parent_directory, label_directory)
		output_label_directory = os.path.join(output_directory, label_directory)  # Create an output directory for this label
		print(output_label_directory)
	    # Ensure the output label directory exists, or create it
		if not os.path.exists(output_label_directory):
			os.makedirs(output_label_directory)
			print(output_label_directory)

		# Filter and collect image paths from the label directory
		image_paths = sc.parallelize(os.listdir(label_path)) \
		.filter(lambda filename: filename.lower().endswith((".jpg", ".JPG"))) \
		.map(lambda filename: os.path.join(label_path, filename))

		# Split the list of image paths randomly into K subsets
		image_paths_subset = image_paths.randomSplit([1.0 / K] * K)

		# Initialize silhouette_coefficients list
		silhouette_coefficients = []

		# Apply K-Means clustering to each subset in parallel and collect the results
		for subset in image_paths_subset:
			subset_segmented = subset.map(lambda path: k_means_clustering(path, k_clusters))

			subset_segmented.foreach(lambda x: (save_segmented_image(x[0], x[1], output_label_directory)))

			# Calculate silhouette scores and store them
			silhouette_scores_subset = subset_segmented.map(lambda x: x[2]).collect()
			silhouette_coefficients.extend(silhouette_scores_subset)

		# Print the average silhouette score for this label
		average_silhouette_score = sum(silhouette_coefficients) / len(silhouette_coefficients)
		print(f"Label: {label_directory}, Average Silhouette Score: {average_silhouette_score}")
		print(f"K-Means clustering and image segmentation completed in {datetime.datetime.now() - start_time}")

	end_time = datetime.datetime.now()
	execution_time = end_time - start_time
	print(f"K-Means clustering and image segmentation completed in {execution_time}")


	

	
	
	
	
	
	
	
	
	
	
	
