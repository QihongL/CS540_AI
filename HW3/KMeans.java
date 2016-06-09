///////////////////////////////////////////////////////////////////////////////
//ALL STUDENTS COMPLETE THESE SECTIONS
//Title:            K-Means cluster algorihtm  
//Files:            KMeansResult.java, HW3.java 
//Semester:         CS540_Intro to AI: Spring 2016
//
//Author:           Qihong Lu
//Email:            qlu36@wisc.edu
//CS Login:         qihong 
//Lecturer's Name:  Collin Engstrom  
///////////////////////////////////////////////////////////////////////////////
import java.util.ArrayList;

public class KMeans {
	/**
	 * The K-means clustering algorithm
	 * 
	 * @param centroids: an 2D array of clustering center
	 * @param instances: an arary of data points 
	 * @param threshold: stopping criterion error bound   
	 * @return result 
	 */
	public KMeansResult cluster(double[][] centroids, double[][] instances, 
			double threshold) {
		if (centroids.equals(null) || instances.equals(null))
			throw new IllegalArgumentException();
		// check feature dimensions? 
		
		// assume the input matrix is square 
		int numInstances = instances.length;
		int numCentroids = centroids.length;
		ArrayList<Double> distortions = new ArrayList<Double>();
		int [] clusterAssignment = new int [numInstances];

		// start main loop of the k-means algorithm
		for(int iter = 0; ; iter ++){
			/** 1. centroids assignment step*/
			int orphanIdx;
			do{
				// assign each instance to the nearest centroids
				clusterAssignment = centroidAssignment(centroids, instances);
				
				// check orphaned centroid 
				orphanIdx = 
						detectOrphanedCentroid(clusterAssignment, numCentroids);
				if (orphanIdx != -1){
					// move the orphaned centroid to the farthest instance
					centroids[orphanIdx] = 
							findFarthestPoint(centroids[orphanIdx],instances);
				}
			}while (orphanIdx != -1); // keep iterate until no orphaned 

			/** 2. centroids moving step (move towards cluster center)*/
			centroids = computeNewCentroids(instances, clusterAssignment, 
					numCentroids);
			
			/** check stopping criterion*/
			// compute the distortion (that controls termination)
			distortions.add(computeDistortion(centroids,
											instances,clusterAssignment));
			// stop if successive distortion is below threshold
			if (iter > 0){
				double errorRatio = computeErrorRatio(distortions);
				// check stopping criterion
				if (errorRatio < threshold) break; 
			}
		}

		// save the results 
		KMeansResult result = new KMeansResult();
		result.clusterAssignment = clusterAssignment;
		result.centroids = centroids;
		result.distortionIterations = new double[distortions.size()];
		for (int iter = 0; iter < distortions.size(); iter ++){
			result.distortionIterations[iter] = distortions.get(iter);
		}
		return result;
	}


	/**
	 * Compute the relative ratio of successive distortion score
	 * This determines the termination of the program
	 * @param distortions
	 * @return the error ratio 
	 */
	private double computeErrorRatio(ArrayList<Double> distortions) {
		if (distortions.equals(null)) throw new IllegalArgumentException();
		// compute successive difference
		double difference = distortions.get(distortions.size()-1) - 
				distortions.get(distortions.size()-2);
		// compute successive ratio
		double errorRatio = 
				Math.abs(difference / distortions.get(distortions.size()-2));
		return errorRatio;
	}

	/**
	 * Compute the distortion scores of all instances with respect to the all 
	 * corresponding centroids
	 * @param centroids: 2d array (numCluster x feature dimensions)
	 * @param instances: 2d array (numInstances x feature dimensions)
	 * @param clusterAssignment 1d array (numInstances x 1)
	 * @return distortion scores 
	 */
	private Double computeDistortion(double[][] centroids,
			double[][] instances, int[] clusterAssignment) {
		if (centroids.equals(null) || instances.equals(null) 
				|| clusterAssignment.equals(null))
			throw new IllegalArgumentException();
		
		// get dimensions
		int numInstances = instances.length;
		int numCentroids = centroids.length;
		double distortion = 0; 
		// loop over all centroids
		for (int cc = 0; cc < numCentroids; cc ++){
			// compute the distortion score for each centroid 
			double partialDistortion = 0; 
			for (int m = 0; m < numInstances; m ++){
				// if this instance belongs to the current cluster, sum it 
				if (clusterAssignment[m] == cc){
					double tempDist = 
							computeDistance(instances[m], centroids[cc]);
					partialDistortion += Math.pow(tempDist, 2);
				}
			}
			// accumulate overall distortion from all centroids 
			distortion += partialDistortion;
		}
		return distortion;
	}


	/**
	 * Compute the location of new centroids
	 * @param instances: 2d array (numInstances x feature dimensions)
	 * @param clusterAssignment: 1d array (numIstances x 1)
	 * @param numCentroids: an interger 
	 * @return the new centroid location (numCentroids x feature dimensions)
	 */
	private double[][] computeNewCentroids(double[][] instances, 
			int[] clusterAssignment, int numCentroids) {
		if (instances.equals(null) || clusterAssignment.equals(null))
			throw new IllegalArgumentException();
		int numInstances = instances.length;
		int numFeatures = instances[0].length;
		double [] [] newCentroids = new double [numCentroids][numFeatures];
		// compute the new centroids location for each cluster
		for (int cc = 0; cc < numCentroids; cc ++){
			double [] tempSum = new double[numFeatures];
			// loop over all instances
			int clusterSize = 0;
			for (int m = 0; m < numInstances; m ++){
				// if this data point is assigned to current cluster 
				if (clusterAssignment[m] == cc){
					// loop over all dimensions to compute average distances
					tempSum = vectorAddition(tempSum,instances[m],numFeatures);
					clusterSize++;
				}
			}
			// compute new center location by averging tempSum
			newCentroids[cc] = scalarMultiplication(tempSum, 1.0/clusterSize);
		}
		return newCentroids;
	}

	/**
	 * Get the assignment vector, where each component indicate the centroid
	 * assignment corresponds to that particular instance
	 * @param centroids: 2d array (numCluster x feature dimensions)
	 * @param instances: 2d array (numInstances x feature dimensions)
	 * @return assignment as a 1d array
	 */
	private int[] centroidAssignment(double[][]centroids, double[][]instances){
		if (centroids.equals(null) || instances.equals(null))
			throw new IllegalArgumentException();
		int numInstances = instances.length;
		int numCentroids = centroids.length;
		// preallocate
		int [] assignmentVector = new int [numInstances];
		// loop over all instances
		for (int m = 0; m < numInstances; m ++){	 
			// loop over all centroids (to find the nearest centroids)
			ArrayList<Double> distances = new ArrayList<Double>();
			for (int c = 0; c < numCentroids; c ++){
				// compute the distance(current point -> current centroids)
				double curDistance = 
						computeDistance(instances[m], centroids[c]);
				// add it to the distances list
				distances.add(curDistance);
			}// end of centrods-loop

			// assign it to corresponding centroids
			double minDistance = getMin(distances);
			assignmentVector[m] = distances.indexOf(minDistance);					
		}// end of instances-loop
		return assignmentVector;
	}

	/**
	 * given a centroid and all instances, find the farthest instance from 
	 * the centroid 
	 * @param centroid (a vector of coordinates)
	 * @param instances a instance matrix (numInstances by features)
	 * @return the farthest instance
	 */
	private double[] findFarthestPoint(double[] centroid, double[][]instances){
		if (centroid.equals(null) || instances.equals(null))
			throw new IllegalArgumentException();
		
		double maxDistance = Double.NEGATIVE_INFINITY;
		double tempDistance; 
		int maxIndex = -1;	// because we have a finite set, maximum exists 
		// check all instances
		for(int m = 0; m < instances.length; m ++){
			tempDistance = computeDistance(centroid, instances[m]);
			// if bigger distance is found, save the distance and the its index 
			if (maxDistance < tempDistance){
				maxDistance = tempDistance;
				maxIndex = m; 
			}
		}
		return instances[maxIndex];
	}

	/**
	 * Detect if there is any orphaned centroid 
	 * 
	 * @param clusterAssignment indicates the assignment for all instances
	 * @param numCentroids	the number of centroids, or k
	 * @return the index of orphaned centroids, return -1 if no orphan   
	 */
	private int detectOrphanedCentroid(int [] clusterAssignment, 
			int numCentroids) {
		if (clusterAssignment.equals(null)) 
			throw new IllegalArgumentException();
		// loop over all centroids
		for (int cc = 0; cc < numCentroids; cc ++){
			boolean isOrphan = true; 
			for (int m = 0; m < clusterAssignment.length; m ++){
				if (clusterAssignment[m] == cc){
					isOrphan = false; 
					break; 
				}
			}
			if (isOrphan)  return cc;
		}
		return -1;
	}

	/**
	 * Multiply a vector by a number, component-wise 
	 * @param x: the input vector
	 * @param scalar: the number 
	 * @return the result
	 */
	private double[] scalarMultiplication(double[] x, double scalar){
		if (x.equals(null)) throw new IllegalArgumentException();
		double [] result = new double [x.length];
		// multiply term by term 
		for (int n = 0; n < x.length; n ++){
			result[n] = x[n] * scalar;
		}
		return result;
	}

	/**
	 * Compute component-wised sum of two vectors in n dimensional space
	 * @param x, y: two input vectors
	 * @return sum: the sum of x and y 
	 */
	private double[] vectorAddition(double [] x, double [] y, int dimension){
		if (x.length != y.length || x.equals(null) || y.equals(null)) 
			throw new IllegalArgumentException();
		// sum the two vector component wise
		double [] sum = new double[x.length]; 
		for (int n = 0; n < dimension; n ++){
			sum[n] = x[n] + y[n];
		}
		return sum; 
	}

	/**
	 * Find the minimal entry within a vector 
	 * @param vector
	 * @return the minimal component
	 */
	private double getMin(ArrayList<Double> vector){
		if (vector.equals(null)) throw new IllegalArgumentException();
		double minimum = Double.POSITIVE_INFINITY;
		for (int i = 0; i < vector.size(); i ++){
			if (vector.get(i) < minimum) minimum = vector.get(i);
		}
		return minimum;
	}

	/**
	 * Compute the Euclidean distance of two vectors 
	 * @param x, y: two vectors with the same dimension
	 * @return distance from x to y
	 */
	private double computeDistance(double[] x, double [] y){
		if (x.length != y.length || x.equals(null) || y.equals(null)) 
			throw new IllegalArgumentException();
		// sum over all coordinates
		double tempSum = 0;
		for (int n = 0; n < x.length; n++){
			tempSum += Math.pow((x[n] - y[n]), 2);
		}
		// take square root 
		double distance = Math.sqrt(tempSum);
		return distance;
	} 

}// end of definition of the class