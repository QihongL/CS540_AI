///////////////////////////////////////////////////////////////////////////////
// Title: 			 CS540 - HW5-4 
// Project:          Naive Bayes spam filter  
// Files:            ClassifyResult.java, NaiveBayesClassifier.java, HW5.java, 
// 					 Instance.java, Label.java   
// Semester:         Spring 2016
// Lecturer's Name:  Collin Engstrom
//
// Author:           Qihong Lu
// Email:            qihong.lu@wisc.edu
// CS Login:         qihong
// WISC-ID: 		 qlu36
///////////////////////////////////////////////////////////////////////////////
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {

	private int numWordType_total; // |word types| in train+test set
	private Map<String, Integer> wordType_freq_spam = new HashMap<String, Integer>();
	private Map<String, Integer> wordType_freq_ham = new HashMap<String, Integer>();
	private Set<String> allWordTypes_train = new HashSet<String>();
	private double p_spam;
	private double p_ham; 
	private int totalCount_spam = 0;
	private int totalCount_ham = 0;
	// constant 
	private double DELTA = 0.00001;

	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {
		if (trainingData.equals(null)) throw new IllegalArgumentException();
		// save the number of "word type" in training set to a field
		// as this will be used later when doing add-delta smoothing  
		numWordType_total = v; 
		// get a list of all word types
		getAllWordTypes(trainingData);
		// compute the priors of the labels
		p_spam = computeLabelPriors(trainingData, Label.SPAM);
		p_ham = computeLabelPriors(trainingData, Label.HAM);
		// compute P(W=w| Label = l) for all w and l 
		computeWordsFreqGivenLabel(trainingData, allWordTypes_train);
	}


	/**
	 * Returns the prior probability of the label parameter 
	 * i.e. P(SPAM) or P(HAM)
	 * 
	 * @param label: the label value
	 * @return P(label)  
	 */
	@Override
	public double p_l(Label label) {
		if (label.equals(Label.SPAM)){
			// return P(spam)
			return p_spam;
		} else if (label.equals(Label.HAM)) {
			// return P(ham)
			return p_ham;
		} else {
			throw new IllegalArgumentException();
		} 
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		if ((!label.equals(Label.SPAM) && !label.equals(Label.HAM)) || 
				word.equals(null)) throw new IllegalArgumentException();
		double numerator;
		double denominator; 
		// compute P(word | label) with add-delta smoothing
		// numerator = frequency(word|label) + delta
		// denominator = v * delta + sum(frequency(word|label)) for all words
		// the case for label = spam 
		if (label.equals(Label.SPAM)){
			if (wordType_freq_spam.containsKey(word)){
				numerator = wordType_freq_spam.get(word) + DELTA;
			} else {  
				numerator = DELTA;
			}
			denominator = numWordType_total * DELTA + totalCount_spam;
			// the case for label = ham
		} else if (label.equals(Label.HAM)){
			if (wordType_freq_ham.containsKey(word)) {
				numerator = wordType_freq_ham.get(word) + DELTA;
			} else {  
				numerator = DELTA;
			}
			denominator = numWordType_total * DELTA + totalCount_ham;	
		} else {
			throw new IllegalArgumentException();
		} 
		//compute P(word | label)
		double p_word_label = numerator / denominator;
		return p_word_label;
	}

	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		if (words.equals(null))  throw new IllegalArgumentException();
		// initialize the result 
		ClassifyResult result = new ClassifyResult();
		result.log_prob_spam = Math.log(p_l(Label.SPAM));
		result.log_prob_ham = Math.log(p_l(Label.HAM));
		// check all words in the input message 
		for (int i = 0; i < words.length; i ++){
			// accumulate the log probability values 
			result.log_prob_spam += Math.log(p_w_given_l(words[i], Label.SPAM)); 
			result.log_prob_ham  += Math.log(p_w_given_l(words[i], Label.HAM));
		}
		// set the result label to be the label with bigger log probability 
		if (result.log_prob_spam > result.log_prob_ham){
			result.label = Label.SPAM;
		} else {
			result.label = Label.HAM;
		}

		return result;
	}



	/*************************************************************************/
	/*************************** HELPER FUNCTIONS  ***************************/
	/*************************************************************************/

	/**
	 * Compute the priors of the label values
	 * @param trainingData
	 * @param label
	 * @return P(label)
	 */
	private double computeLabelPriors(Instance[] trainingData, Label label) {
		if (trainingData.equals(null) || label.equals(null))
			throw new IllegalArgumentException();
		int counts = 0; 
		// compute the frequency of the input label 
		for (int i = 0; i < trainingData.length; i ++){
			if (trainingData[i].label.equals(label))
				counts ++;
		}
		// get the probability 
		double probability_label = counts * 1.0 / trainingData.length;
		return probability_label;
	}


	/**
	 * Compute the frequency of a given word, conditioned on all possible
	 * label values  
	 * @param data
	 * @param allWordTypes
	 */
	private void computeWordsFreqGivenLabel(Instance[] data, 
			Set<String> allWordTypes){
		if (data.equals(null)) throw new IllegalArgumentException();		
		// loop over all word types
		Iterator<String> iterator = allWordTypes.iterator();
		while(iterator.hasNext()){
			String curWordType = iterator.next();
			int count_spam = 0;
			int count_ham = 0; 			
			// loop over all messages 
			for (int m = 0; m < data.length; m ++){
				// increment the counter for the w.r.t the given label 
				if (data[m].label.equals(Label.SPAM)){
					count_spam += wordOccurances(curWordType, data[m].words); 
				} else if (data[m].label.equals(Label.HAM)) {
					count_ham += wordOccurances(curWordType, data[m].words);
				} else {
					throw new IllegalArgumentException("Exists mislabeled data");
				}
			} 
			// save the results 
			wordType_freq_spam.put(curWordType, count_spam);
			wordType_freq_ham.put(curWordType, count_ham);
			totalCount_spam += count_spam;
			totalCount_ham  += count_ham;
		}
	}


	/**
	 * Get the occurances of a word in a message
	 * @param wordType
	 * @param message
	 * @return the occurances
	 */
	private int wordOccurances(String wordType, String[] message) {
		// TODO Auto-generated method stub
		if (message.equals(null)) throw new IllegalArgumentException();
		int count = 0; 
		// loop over all words in the message 
		for (int i = 0; i < message.length; i ++){
			// if the input wordType matches any word in the message 
			if (wordType.equals(message[i])) 
				// then the word is in the message
				count ++;  
		}
		return count; 
	}

	
	/**
	 * Construct the set of all word types in the training data.  
	 * @param trainingData
	 */
	private void getAllWordTypes (Instance[] trainingData){
		if (trainingData.equals(null)) throw new IllegalArgumentException();
		// loop over all messages 
		for (int m = 0; m < trainingData.length; m ++) {
			// for each message, loop over all words in the message
			for (int w = 0; w < trainingData[m].words.length; w++){
				// add words to the set of all wordTypes
				allWordTypes_train.add(trainingData[m].words[w]);
			}
		}
	}

} // end of definition of the class
