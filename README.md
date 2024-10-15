# Soybean-Leaf-Disease-Detection
Develop and implement an accurate and efficient deep learning model (SoyNet DNN) for Soybean leaf disease detection, ensuring early diagnosis and timely intervention to protect crop health
Implemented resilient image preprocessing techniques to extract essential phenotypic information from Soybean leaf images. This step ensures high-quality data for accurate disease detection and facilitates feature extraction for deep-learning models.
Optimize and enhance proposed methods (scalable K-means) using scalability by integrating Apache Spark into the existing system. Which facilitates faster and more parallelized processing of large datasets, and improves overall system efficiency.

Algorithm 1: K-means ()
Input: data, k, max iters
Output: centroids, labels
centroids ← data[np.random.choice(range(len(data)), k, replace =False)]
for iteration in max iters do
distances ← np.linalg.norm(data[:, np.newaxis] − centroids, axis = 2);
labels ← np.argmin(distances, axis = 1);
new centroids ← [];
for i in range(k) do
if np.any(labels == i) then
new centroids.append(data[labels == i].mean(axis = 0));
else
new centroids.append(data[np.random.choice(len(data))]);
new centroids ← np.array(new centroids);
if np.all(new centroids == centroids) then
break;
centroids ← new centroids;
return centroids, labels;

Algorithm 2: Scalable K-means for image segmentation
Input: data, k, max iters
Output: centroids, labels
Perform image partitions on images using the Parallelize() function
from Algorithm 3.
Perform K-means using Algorithm 1.
ImagePartitions = parallelize(image data, num partitions);
for partition in ImagePartitions do
for image subset in partition do
DataSubsets = DivideDataIntoKSubsets(image subset, k);
KMeansResults = DataSubsets.map(kmeans(data, k,max iters));
for kmeans result in KMeansResults do
CalculateSilhouetteScores(kmeans result);
SaveSegmentedImages(kmeans result);

Algorithm 3: Parallelize ()
Input: data, numSlices
Output: RDD
X ← data // Assume X is the array of data samples:
X = {x1, x2, x3, . . . };
Split X into partitions P such that X = {x1, x2, x3, . . . } is divided into P partitions: {x1, x2}, {x3, x4}, . . . ;
DistributePartitions(P )
AssignPartitionsToNodes(P )
return RDD

Algorithm 4: Map ()
Input: K-means()
Output: new RDD
for each element xi in RDD do
xi transf ormed ← ApplyFunction(xi, K-means()) // Apply function F to xi and transform xi using F ;
CreateNewRDD(xi transf ormed) // Generate new RDD with transformed elements;
end
return new RDD
