import java.util.ArrayList;


public class Testing {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int test = 0; 
		do {
			test ++; 
		}while (test < 3);
		System.out.println();
		System.out.println(test);
		System.out.println();
		
	}

	

	
	/**
	 * Print a vector, for debugging purpose
	 * @param x: the input vector
	 * @param description: the description of what the vector is 
	 */
	private void printVector(double[] x, String description){
		System.out.print(description);
		for(int n = 0 ; n < x.length; n ++){
			System.out.print(x[n]);
			System.out.print(' ');
		}
		System.out.println();
	}


	/**
	 * Print a vector, for debugging purpose
	 * @param x: the input vector
	 * @param description: the description of what the vector is 
	 */
	private void printVector(int[] x, String description){
		System.out.print(description);
		for(int n = 0 ; n < x.length; n ++){
			System.out.print(x[n]);
			System.out.print(' ');
		}
		System.out.println();
	}	

	


	private void printDim (int numInstances, int numFeatures, int numCentroids){
		// print check dim 
		System.out.print("Input centroids: ");
		System.out.print(numCentroids);
		System.out.print(" by ");
		System.out.println(numFeatures);

		System.out.print("Input data matrix: ");
		System.out.print(numInstances);
		System.out.print(" by ");
		System.out.println(numFeatures);
	}

	

	/**
	 * find the maximum of a vector. 
	 * @param vector
	 * @return max 
	 */
	private double getMax(double [] vector){
		if (vector.equals(null)) throw new IllegalArgumentException();

		double maximum = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < vector.length; i ++){
			if (vector[i] > maximum) maximum = vector[i];
		}
		return maximum;
	}


	private double getMax(ArrayList<Double> vector){
		double maximum = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < vector.size(); i ++){
			if (vector.get(i) > maximum) maximum = vector.get(i);
		}
		return maximum;
	}


	/**
	 * find the minimum of a vector. 
	 * @param vector
	 * @return min 
	 */
	private double getMin(double [] vector){
		if (vector.equals(null)) throw new IllegalArgumentException();

		double minimum = Double.POSITIVE_INFINITY;
		for (int i = 0; i < vector.length; i ++){
			if (vector[i] < minimum) minimum = vector[i];
		}
		return minimum;
	}

}
