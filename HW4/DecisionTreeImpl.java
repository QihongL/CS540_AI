///////////////////////////////////////////////////////////////////////////////
//                   ALL STUDENTS COMPLETE THESE SECTIONS
// Title:            Decision tree, the ID3 algorithm 
// Files:            DataSet.java, DecisionTree.java, DecTreeNode.java,  
// 					instance.java, HW4.java
// Semester:         Spring 2016
//
// Author:           Qihong Lu
// CS Login:         qihong
// Lecturer's Name:  Collin Engstrom
//////////////////////////// 80 columns wide //////////////////////////////////
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
	// discrete values taken
	// by attributes

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}


	/**
	 * Build a decision tree given only a training set.
	 * 
	 * I will use conventional ML notation, where X stands for the design 
	 * matrix, y is a column vectors with class labels for all instances. 
	 * There are M training examples and N features (attributes) in this data 
	 * set and I will use m for the running indices for training examples, 
	 * and n for the running indices for the features.    
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {
		if (train.equals(null)) throw new IllegalArgumentException();
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// transform the data into matrix form
		int [][] X = formDataMatrix(train.instances, attributes);
		int [] y = getLabelVector(train.instances);
		// the root node has parent attribute value of -1. 
		int parentAttributeVal = -1;
		// grow a tree from the root
		root = treeBuilding(X, y, parentAttributeVal, labels, attributes, 
				attributeValues, attributes);
	}


	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		this(train);
		if (train.equals(null) || tune.equals(null)) 
			throw new IllegalArgumentException();
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		// compute the accuracy with the full tree
		double baseAccuracy = getAccuracy(tune.instances);
		// truncate the tree
		while(true){
			// get the references for all nodes 
			ArrayList<DecTreeNode> allNodes = 
					getAllTreeNodes(new ArrayList<DecTreeNode>(), root);
			// compute accuracy when different node was pruned   
			double [] accuracies = getAccuracies(baseAccuracy, allNodes, 
					tune.instances);			
			// end of loop if deleting any node decrease accuracy
			if (getMax(accuracies) < baseAccuracy){
				break;
			} else {
				// choose the pruning-node that improves accuracy the most  
				int maxIndex = getIndex(accuracies, getMax(accuracies));
				// delete a node
				allNodes.get(maxIndex).terminal = true;
			}
		}
	}


	@Override
	public String classify(Instance instance) {
		if (instance.equals(null)) 
			throw new IllegalArgumentException();
		// figure out which attribute to use 
		int curAttributeIdx = root.attribute;
		// figure out which child to choose
		int childIdx = instance.attributes.get(curAttributeIdx);
		// if no child, then return the default class label
		if (root.terminal){
			return Integer.toString(root.label);
		}
		// if child exists, going down the tree
		DecTreeNode curNode = root.children.get(childIdx);
		int labelIdx = getLeafLabel(instance, curNode);
		return Integer.toString(labelIdx);
	}


	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else{
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	/**
	 * Print the mutual information of each attribute at the root node
	 */
	@Override
	public void rootInfoGain(DataSet train) {
		if (train.equals(null)) 
			throw new IllegalArgumentException();
		this.labels = train.labels; // the range of the label values
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// transform the data into matrix form
		int [][] X = formDataMatrix(train.instances, attributes);
		int [] y = getLabelVector(train.instances);
		// get entropy values
		double IG[] = getInfomationGain(X, y, attributes, attributeValues);
		// print the information gains to console  
		for (int i = 0; i < IG.length; i ++)
			System.out.format("%s %.5f\n", train.attributes.get(i), IG[i]);
	}



	////////////////////////////////////////////////////////////////////	
	//////////////////// OTHER HELPER FUNCTIONS ////////////////////////
	////////////////////////////////////////////////////////////////////

	/**
	 * Build a decision tree recursively.
	 *  
	 * @param X: the design matrix 
	 * @param y: the class labels
	 * @param parentAttributeVal: the attribute value of the parent
	 * @param labels: the range of the labels (y)
	 * @param attributes: the features
	 * @param attributeValues: the range of each feature
	 * @return a decision tree as a mapping from X to y 
	 */
	private DecTreeNode treeBuilding(int[][] X,int[] y, int parentAttributeVal, 
			List<String> labels, List<String> inputAttributes, 
			Map<String, List<String>> attributeValues, 
			List<String> rawAttributes) {
		DecTreeNode node = null;
		// check some strange cases
		if (y == null) {
			// use the default label 
			return new DecTreeNode(0, 0, parentAttributeVal, true);
			// return new DecTreeNode(0, null, parentAttributeVal, true);
		} else if  (classIsPure(y)){
			// use the 1st label 
			return new DecTreeNode(y[0], 0, parentAttributeVal, true);
		} else if (Integer.valueOf(inputAttributes.size()).equals(0)){
			// return majority class: 
			return new DecTreeNode(getMajorityClass(y, labels), 0, 
					parentAttributeVal, true);
		} else {
			/** the non-trivial decision tree case starts here */
			// 1. create the parent node 
			// get the current majority class
			int majorityClass = getMajorityClass(y, labels);
			// get the feature with the max information gain 
			double [] infoGain = getInfomationGain(X, y, 
					inputAttributes, attributeValues);
			// get the (zero-based) index for the feature with max IG
			int maxIGFeatureIdx = getIndex(infoGain, getMax(infoGain));
			// compute the feature index using the original feature list 
			String bestAttribute = inputAttributes.get(maxIGFeatureIdx);
			int rawMaxIGFeatureIdx = rawAttributes.indexOf(bestAttribute);
			// check if the node is terminal
			boolean isTerminal = false; 
			if (classIsPure(y) || inputAttributes.isEmpty()) 
				isTerminal = true; 
			// create the root node! 
			node = new DecTreeNode(majorityClass, rawMaxIGFeatureIdx, 
					parentAttributeVal, isTerminal);
			// 2. gather information for creating the children
			// boolVec that indicates the presence of 
			// (n-th) feature value of the m-th instance
			int numVals = attributeValues.get(inputAttributes.get(maxIGFeatureIdx)).size();
			boolean [][] maxIGF_valIdx = getIndicatorVectors(X, maxIGFeatureIdx, 
					numVals);
			// attributes and their attribute values that the child can access
			List<String> attributes_child = new ArrayList<String>(inputAttributes);
			attributes_child.remove(maxIGFeatureIdx);
			Map<String,List<String>> attributeValues_child = new HashMap<String, 
					List<String>>(attributeValues);
			attributeValues_child.remove(inputAttributes.get(maxIGFeatureIdx));
			// 3. create the children 
			// split the tree according to the max IG feature
			for (int val = 0; val < numVals; val++){
				// truncate the data set and the label vector
				int[] subsetY = getSubsetLabels(y, maxIGF_valIdx[val]);
				int[][] subsetX = getSubsetData(X, maxIGF_valIdx[val], 
						maxIGFeatureIdx);
				int[][] subsetXD = deleteFeature(subsetX, maxIGFeatureIdx);
				// create the children recursively
				node.addChild(treeBuilding(subsetXD, subsetY, val, labels, 
						attributes_child, attributeValues_child, attributes));
			}
		} // end of handling non-trivial node 
		return node; 
	} // end of the tree building method


	/**
	 * compute the accuracy while deleting different node
	 * @param baseAccuracy: the accuracy before deleting any node 
	 * @param allNodes
	 * @param instances
	 * @return the accuracy while deleting different node
	 */
	private double[] getAccuracies(double baseAccuracy, 
			ArrayList<DecTreeNode> allNodes, List<Instance> instances) {
		if (instances.equals(null) || allNodes.equals(null) 
				|| baseAccuracy < 0 || baseAccuracy > 1) 
			throw new IllegalArgumentException();
		double [] accuracies = new double[allNodes.size()];
		// compute the accuracy while deleting different node 
		for (int i = 0; i < allNodes.size(); i++){
			// if is terminal, then no deletion can be made 
			if (allNodes.get(i).terminal){
				continue; 
			} else { 
				allNodes.get(i).terminal = !allNodes.get(i).terminal;
				// compute the accuracy while deleting a node 
				accuracies[i] = getAccuracy(instances);
				allNodes.get(i).terminal = !allNodes.get(i).terminal;	
			}
		}
		return accuracies;
	}


	/**
	 * Recursively find the label of the leaf node corresponds to the instance
	 * attributes 
	 * @param instance
	 * @param treeNode
	 * @return the label 
	 */
	private int getLeafLabel(Instance instance, DecTreeNode treeNode){
		if (instance.equals(null) || treeNode.equals(null))
			throw new IllegalArgumentException(); 
		// figure out which feature for the current step 
		int curAttributeIdx = treeNode.attribute;
		// figure out which child to choose 
		int childIdx = instance.attributes.get(curAttributeIdx);
		if (treeNode.terminal == true)
			return treeNode.label;
		return getLeafLabel(instance, treeNode.children.get(childIdx));
	}

	/**
	 * Compute the classification accuracy of the root 
	 * accuracy
	 * @param instances: a list of instances
	 * @return the classification accuracy
	 */
	private double getAccuracy(List<Instance> instances) {
		if (instances.equals(null)) 
			throw new IllegalArgumentException();
		// compute the hit vector
		int hitCount = 0; 
		for (int m = 0; m < instances.size(); m ++){
			int treeOutput = Integer.parseInt(classify(instances.get(m)));
			// if the m-th prediction is correct, m-th entry sets to "true"
			if (Integer.valueOf(treeOutput).equals(instances.get(m).label)){
				hitCount ++; 
			}
		}
		// compute accuracy 
		double accuracy = hitCount * 1.0 /  instances.size();
		return accuracy;
	}

	/**
	 * save all decision tree nodes into an list
	 * @param allTreeNodes
	 * @param treeNode
	 * @return an array list of tree nodes 
	 */
	private ArrayList<DecTreeNode> getAllTreeNodes (
			ArrayList<DecTreeNode> nodeList, DecTreeNode treeNode){
		if (treeNode.equals(null)) 
			throw new IllegalArgumentException();
		// add the node to the list 
		nodeList.add(treeNode);
		// recursively traverse the child node
		for (int i = 0; i < treeNode.children.size(); i ++)
			if (!treeNode.children.get(i).terminal)
				getAllTreeNodes(nodeList, treeNode.children.get(i));
		return nodeList; 
	}

	/**
	 * Compute the classification accuracy of the root 
	 * accuracy
	 * @param instances: a list of instances
	 * @return the classification accuracy
	 */
	private double getAccuracy(DecTreeNode tree, List<Instance> instances) {
		if (tree.equals(null) || instances.equals(null)) 
			throw new IllegalArgumentException();
		// compute the hit vector
		boolean [] hits = new boolean[instances.size()];
		for (int m = 0; m < instances.size(); m ++){
			String treeOutput_string = treeClassify(tree, instances.get(m));
			int treeOutput = Integer.parseInt(treeOutput_string);
			// if the m-th prediction is correct, m-th entry sets to "true"
			if (Integer.compare(treeOutput, instances.get(m).label) == 0){
				hits[m] = true; 
			}
		}
		// compute the number of correct prediction
		int numHits = countNumberOfTruth(hits);
		// compute accuracy 
		double accuracy = numHits * 1.0 /  instances.size();
		return accuracy;
	}

	/**
	 * Compute the classification output of a given tree 
	 * @param tree
	 * @param instance
	 * @return the output class
	 */
	private String treeClassify(DecTreeNode tree, Instance instance) {
		if (instance.equals(null)|| tree.equals(null)) 
			throw new IllegalArgumentException();
		// figure out which attribute to use 
		int curAttributeIdx = tree.attribute;
		// figure out which child to choose
		int childIdx = instance.attributes.get(curAttributeIdx);
		// if no child, then return the default class label
		if (tree.children.size() == 0)
			return Integer.toString(tree.label);
		// if child exists, going down the tree
		DecTreeNode curNode = tree.children.get(childIdx);
		int labelIdx = getLeafLabel(instance, curNode);
		return Integer.toString(labelIdx);
	}


	/**
	 * Make a copy of the input decision tree 
	 * @param tree
	 * @return the copy of the input tree
	 */
	private DecTreeNode copyTree (DecTreeNode tree){
		if (tree.equals(null)) 
			throw new IllegalArgumentException(); 
		// initialize the copy 
		DecTreeNode copy = new DecTreeNode(tree.label, tree.attribute, 
				tree.parentAttributeValue, tree.terminal);
		// copy the children
		if (!tree.terminal){
			for (int i = 0; i < tree.children.size(); i ++){
				copy.addChild(copyTree(tree.children.get(i)));
			}
		}
		return copy; 
	}

	/**
	 * Get a subset of the data (with a particular feature value for the 
	 * input feature), and delete the input feature.  
	 * @param X: the initial data matrix
	 * @param featureValueIndex: the index for the selected feature
	 * @param deletedFeatureIdx: the feature that you want to delete
	 * @return the subset data matrix
	 */
	private int [][] getSubsetData(int[][] X, boolean[] featureValIdx, 
			int deletedFeatureIdx) {
		if (X.equals(null) || featureValIdx.equals(null)) 
			throw new IllegalArgumentException();
		// count number of instances of X with a particular feature value
		int numInstances = countNumberOfTruth(featureValIdx);   
		// initialize the subset of the full data matrix
		int [][] subset = new int [X.length][numInstances];
		int curExample = 0; 
		// loop over all instances in X... 
		for (int m = 0; m < X[0].length; m ++){
			// ... to find instances with the given feature value
			if (featureValIdx[m]){
				for (int n = 0; n < X.length; n++){ 
					subset[n][curExample] = X[n][m];
				}
				curExample ++; 
			}
		}
		return subset;
	}


	/**
	 * helper function for "getSubsetData"
	 * Delete a row in 2d array 
	 * @param X: the initial matrix (2d array)
	 * @param deletedFeatureIdx
	 * @return the matrix with the input row deleted 
	 */
	private int[][] deleteFeature(int [][] X, int deletedFeatureIdx) {
		if (X.equals(null) || deletedFeatureIdx < 0) 
			throw new IllegalArgumentException();
		// delete a row of features with index = deletedFeatureIdx
		int [][] deletedSubset = new int [X.length-1][X[0].length];
		int count = 0; 
		for (int n = 0; n < X.length; n ++){
			if (Integer.valueOf(n).equals(deletedFeatureIdx)) 
				continue; 
			for (int m = 0; m < X[0].length; m ++){
				deletedSubset [count][m] = X[n][m];
			}
			count ++; 
		}
		return deletedSubset;
	}


	/**
	 * Get the number of truncated y, the class labels 
	 * @param y: original class labels
	 * @param featureValIdx
	 * @return: truncated y
	 */
	private int[] getSubsetLabels(int[] y, boolean[] featureValIdx) {
		if (y.equals(null) || featureValIdx.equals(null)) 
			throw new IllegalArgumentException();
		// count number of instances of y with a particular feature value
		int numInstances = countNumberOfTruth(featureValIdx);
		if (Integer.valueOf(numInstances).equals(0)) return null; 
		// initialize the subset of the full label vector y
		int [] subsetY = new int [numInstances];
		int curExample = 0; 
		// loop over all instances in y... 
		for (int m = 0; m < y.length; m ++){
			// ... to find instance label with the given feature value
			if (featureValIdx[m]){ 
				subsetY[curExample] = y[m];
				curExample ++; 
			}
		}
		return subsetY;
	}


	/**
	 * Given a list of labels, get all possible values of the list 
	 * @param labels
	 * @return an arraylist of the range of the input list 
	 */
	private ArrayList<Integer> getLabelRange(int[] y) {
		if (y.equals(null)) throw new IllegalArgumentException();
		// loop over all labels
		ArrayList<Integer> labelRange = new ArrayList<Integer>();
		// if detected unseen label, add it to the range of possible labels
		for (int m = 0; m < y.length; m ++){
			if (!labelRange.contains(y[m])) {
				labelRange.add(y[m]);
			}
		}
		return labelRange;
	}


	/**
	 * Given a list of labels, check if all of them are the same
	 * @param labels
	 * @return true if all of them belong to one class, false otherwise 
	 */
	private boolean classIsPure(int[] y) {
		// if there is no instance, then the class is pure 
		if (y.equals(null))  
			return true; 
		ArrayList<Integer> labelRange = getLabelRange(y);
		// the class is pure iff only 1 label was detected 
		if ( Integer.valueOf(labelRange.size()).equals(1)){
			return true;
		}
		return false;
	}

	/**
	 * Compute the information gain for all attributes
	 * @param X: the data matrix (attribute by instances)
	 * @param y: the labels 
	 * @param attributes: the features
	 * @param attributeValues: the range of features
	 * @return the information gain for all attributes
	 */
	private double[] getInfomationGain(int [][] X, int [] y, List<String> attributes, 
			Map<String, List<String>> attributeValues) {
		if (X.equals(null) || y.equals(null) || attributes.equals(null) || 
				attributeValues.equals(null))
			throw new IllegalArgumentException();
		// pre-allocate
		double [] infoGain = new double [attributes.size()];
		// get entropy values
		double totalEntropy = getTotalEntropy(X, y, labels);
		double [] conditonalEntropy = getCondEntropy(X, y, attributes, attributeValues);
		// compute information gains for all features 
		for (int i = 0; i < conditonalEntropy.length; i ++){
			infoGain[i] = totalEntropy - conditonalEntropy[i];
		}
		return infoGain;
	}

	/**
	 * Compute the total entropy: H(Y)
	 * @param X: the data matrix (attribute by instances) 
	 * @param y: the labels
	 * @param labels: the range of labels 
	 * @return totalEnt: H(Y)
	 */
	private double getTotalEntropy(int [][] X, int [] y, List<String> labels) {
		if (X.equals(null) || y.equals(null) || labels.equals(null))
			throw new IllegalArgumentException();
		// get probability distribution of all labels 
		int [] frequencies = computeFrequencies(y, labels);
		double [] pmf_Y = computePMF(frequencies);
		// calculate the entropy given the probability distribution
		double totalEntropy = computeEntropy(pmf_Y);
		return totalEntropy;
	}


	/**
	 * Calculate conditional entropy 
	 * @param train: the training data
	 * @return condEnt: conditional entropy for all features
	 */
	private double[] getCondEntropy(int [][] X, int [] y, 
			List<String> attributes, Map<String,List<String>> attributeValues){
		if (X.equals(null) || y.equals(null) || attributes.equals(null) || 
				attributeValues.equals(null))
			throw new IllegalArgumentException();

		// loop over features (columns)
		double [] condEnt = new double [attributes.size()];
		for (int n = 0; n < attributes.size(); n++) {
			// 1. compute the distribution of feature values
			int numAttVals = attributeValues.get(attributes.get(n)).size();
			int []  featureFreq = new int[numAttVals];
			for (int m = 0; m < y.length; m ++)
				featureFreq[X[n][m]] ++;
			// compute P(X=v), for all v
			double [] featurePmf = computePMF(featureFreq);

			// 2. compute the conditional entropy
			// preallocate
			double [] condEnt_v = new double [numAttVals]; 
			// loop over all possible values of the n-th features
			for (int v = 0; v < numAttVals; v ++){
				ArrayList<Integer> reducedLabels = new ArrayList<Integer>();
				// loop over instances (rows)
				for (int m = 0; m < y.length; m ++){
					// record the value of Y iff X = v
					if ( Integer.valueOf(X[n][m]).equals(v)) 
						reducedLabels.add(y[m]);
				}
				// get the PMF of Y given X = v  
				int [] tempFreq = computeFrequencies(reducedLabels, labels);
				double [] tempPmf = computePMF(tempFreq);
				// conditional entropy of Y given n-th X = f-th value
				// compute H(Y|X=v), for all v
				condEnt_v[v] = computeEntropy(tempPmf);
			}
			// Accumulate information gain for the n-th feature
			// H(Y|X) = sum( P(X=v) * H(Y|X=v) )
			condEnt[n] = vectorSum(elementwiseMultiply(featurePmf,condEnt_v));
		}
		return condEnt;
	}


	/**
	 * Compute the entropy given a probability distribution
	 * log base 2 was used
	 * @param pmf: a vector of probabilities
	 * @return the entropy 
	 */
	private double computeEntropy(double [] pmf) {
		if (pmf.equals(null)) throw new IllegalArgumentException();
		double entropy = 0;
		// accumulate entropy
		for (int i = 0; i < pmf.length; i ++){
			if(pmf[i] > Double.MIN_VALUE) // avoid log(0)
				entropy -= pmf[i] * ( Math.log(pmf[i]) / Math.log(2) );
		}
		return entropy; 
	}


	/**
	 * Given a data set, which contains a bunch of instances, each of which 
	 * is associated with a label. This method counts which labels is the 
	 * majority class.   
	 * @param instances
	 * @param labelRange
	 * @return the index of the majority class
	 */
	private int getMajorityClass(int [] labels, List<String> labelRange) {
		if (labels.equals(null)) throw new IllegalArgumentException();
		int [] frequency = new int [labelRange.size()]; 
		// loop over all instances, count the frequency of each class
		for (int m = 0; m < labels.length; m ++) 
			frequency[labels[m]] ++; 
		// get the class index with the max frequency 
		int majorityClassLabel = getIndex(frequency, getMax(frequency));
		return majorityClassLabel;
	}


	/**
	 * Transform the data into matrix form
	 * @param instances
	 * @param attributes
	 * @return the data matrix
	 */
	private int[][] formDataMatrix(List<Instance> instances, 
			List<String> attributes) {
		if (instances.equals(null) || attributes.equals(null))
			throw new IllegalArgumentException();
		int [][] dataMatrix = new int [attributes.size()][instances.size()];
		// loop over all attributes (columns)
		for (int n = 0; n < attributes.size(); n ++){
			// loop over all instances (rows)
			for (int m = 0; m < instances.size(); m ++){
				dataMatrix[n][m] = instances.get(m).attributes.get(n);
			}
		}
		return dataMatrix;
	}

	/**
	 * Extract the label information from "instances" into a 1D array 
	 * @param instances: contain all the information about all instances 
	 * @return labels_vector: 1D array that contains all the labels
	 */
	private int[] getLabelVector(List<Instance> instances) {
		if (instances.equals(null)) throw new IllegalArgumentException();
		// get labels for all instances 
		int [] labels_vector = new int [instances.size()];
		for (int m = 0; m < instances.size(); m ++){
			labels_vector[m] = instances.get(m).label;
		}
		return labels_vector;
	}


	/**
	 * 
	 * @param instances
	 * @param maxIGFeature
	 * @return
	 */	
	private boolean[][] getIndicatorVectors(int[][] X, int featureIdx, 
			int numAttributeVals) {
		if (X.equals(null)) throw new IllegalArgumentException();

		boolean [][] maxIGFidx = new boolean [numAttributeVals][X[0].length];
		// loop over all possible values of a given attribute 
		for (int attVal = 0; attVal < numAttributeVals ; attVal ++){
			// loop over all instances
			for (int m = 0; m < X[0].length; m ++){
				// if this instance has the current attribute value, take true
				if ( Integer.valueOf(X[featureIdx][m]).equals(attVal)){
					maxIGFidx[attVal][m] = true; 
				} 
			}
		}
		return maxIGFidx;
	}


	/**
	 * Convert the frequency values to a empirical probability mass function 
	 * @param frequencies: counts for all labels
	 * @return PMF: the probability distribution 
	 */
	private double [] computePMF(int [] frequencies){
		if (frequencies.equals(null)) throw new IllegalArgumentException();
		double [] pmf = new double [frequencies.length];
		// get total sum 
		int totalSum = 0;
		for (int i = 0; i < frequencies.length; i ++)
			totalSum += frequencies[i];
		// get probability distribution (= probability mass function) 
		for (int i = 0; i < frequencies.length; i ++)
			pmf[i] = 1.0 * frequencies[i] / totalSum;
		return pmf;  
	}


	/**
	 * Elements in the instances arrayList are labels  (y)
	 * @param instances
	 * @param labels
	 * @return
	 */
	private int [] computeFrequencies(ArrayList<Integer> instances,
			List<String> labelRange) {
		if (instances.equals(null) || labelRange.equals(null))
			throw new IllegalArgumentException();
		int [] frequencies = new int[labelRange.size()];
		for (int m = 0; m < instances.size(); m ++){
			// increment the count for the corresponding index
			frequencies[instances.get(m)] ++;
		}
		return frequencies;
	}

	/**
	 * Compute frequency, for matrix formed data
	 * @param labelVector
	 * @param labelRange
	 * @return frequency counts 
	 */
	private int [] computeFrequencies(int [] labelVector, 
			List<String> labelRange) {
		if (labelVector.equals(null) || labelRange.equals(null)) 
			throw new IllegalArgumentException();
		int [] frequencies = new int[labelRange.size()];
		for (int m = 0; m < labelVector.length; m ++){
			// increment the count for the corresponding index 
			frequencies[labelVector[m]] ++;
		}
		return frequencies;
	}

	/**
	 * Sum a array of numbers
	 * @param x
	 * @return the sum of x_i, for all i 
	 */
	private double vectorSum(double [] x){
		if (x.equals(null)) throw new IllegalArgumentException();
		// accumulate all values in the input vector
		double sum = 0; 
		for (int i = 0; i < x.length; i ++) sum += x[i]; 
		return sum; 
	}

	/**
	 * Compute element-wise Multiplication between 2 vector  
	 * @param x: a input array
	 * @param y: a input array
	 * @return xy: i-th entry is x_i * y_i 
	 */
	private double [] elementwiseMultiply (double [] x, double [] y){
		if (x.length != y.length || x.equals(null) || y.equals(null)) 
			throw new IllegalArgumentException();
		double [] xy = new double [x.length];
		// loop over all dimensions to get xi*yi
		for (int i = 0; i < x.length; i ++){
			xy[i] = x[i] * y[i];
		}
		return xy;
	}

	/**
	 * Get the max of an array of numbers 
	 * @param x: a double array
	 * @return max(x_i) for all i 
	 */
	private double getMax (double [] x){
		if (x.equals(null)) throw new IllegalArgumentException();
		double max = Double.MIN_VALUE;
		// loop over all values. When bigger value is found, update max 
		for (int i = 0; i < x.length; i ++)
			if (x[i] > max) max = x[i];
		return max; 
	}
	/**
	 * Get the max of an array of numbers 
	 * @param x: a integer array
	 * @return max(x_i) for all i 
	 */
	private int getMax(int[] x) {
		if (x.equals(null)) throw new IllegalArgumentException();
		int max = Integer.MIN_VALUE;
		// loop over all values. When bigger value is found, update max 
		for (int i = 0; i < x.length; i ++)
			if (x[i] > max) max = x[i];
		return max; 
	}

	/**
	 * Get the index of a given value in x (the first)
	 * @param x: a double array
	 * @param value: a double value 
	 * @return the index of the input value in the array x (-1 means DNE)
	 */
	private int getIndex (double [] x, double targetValue){
		if (x.equals(null)) throw new IllegalArgumentException();
		int index = -1;
		for (int i = 0; i < x.length; i ++){
			// return the index of the 1st matching value  
			if (Double.valueOf(x[i]).equals(targetValue)){
				return i;   
			}
		}
		return index; 
	}
	/**
	 * Get the index of a given value in x
	 * @param x: integer array
	 * @param value
	 * @return the index of the input value in the array x (-1 means DNE)
	 */
	private int getIndex (int [] x, int targetValue){
		if (x.equals(null)) throw new IllegalArgumentException();
		int index = -1;
		for (int i = 0; i < x.length; i ++){
			// return the index of the 1st matching value
			if (Integer.valueOf(x[i]).equals(targetValue)){
				return i;   
			}
		}
		return index; 
	}

	/**
	 * Given an array of boolean values, count how many of them are "true"
	 * @param x: an array of boolean 
	 * @return 
	 */
	private int countNumberOfTruth(boolean [] x){
		if (x.equals(null)) throw new IllegalArgumentException();
		int numTrueVals = 0;
		// loop over all elements of x 
		for (int i = 0; i < x.length; i ++){
			// increment the count, when x[i] is true 
			if (x[i]) 
				numTrueVals ++;
		}
		return numTrueVals; 
	}

}// end of definition of the class