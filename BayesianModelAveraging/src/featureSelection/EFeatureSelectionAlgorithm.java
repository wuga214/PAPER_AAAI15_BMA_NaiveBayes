package featureSelection;

public enum EFeatureSelectionAlgorithm {

	//rank attributes **********

	CONDITIONAL_ENTROPY_RANK,
	//rank features by conditional entropy

	CORRELATION_BASED_RANK,
	//Evaluates the worth of an attribute by measuring the correlation (Pearson's) between it and the class.
	//LINK: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/CorrelationAttributeEval.html

	GAINRATIO_RANK,
	//Evaluates the worth of an attribute by measuring the gain ratio with respect to the class.
	//GainR(Class, Attribute) = (H(Class) - H(Class | Attribute)) / H(Attribute).
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/GainRatioAttributeEval.html

	INFORMATIONGAIN_RANK,
	//Rank by mutual information between class and feature
	//InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/InfoGainAttributeEval.html

	SYMMETRICAL_UNCERT_RANK,
	//SymmU(Class, Attribute) = 2 * (H(Class) - H(Class | Attribute)) / H(Class) + H(Attribute).
	//LINK: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/SymmetricalUncertAttributeEval.html

	RELIFF,
	//Evaluates the worth of an attribute by repeatedly sampling an instance and considering the value of the given attribute for the nearest instance of the same and different class.
	//Can operate on both discrete and continuous class data.
	//http://weka.sourceforge.net/doc.dev/weka/attributeSelection/ReliefFAttributeEval.html
	
	
	//subset ************

	CORRELATION_BASED_SUBSET,
	//Subsets of features that are highly correlated with the class while having low intercorrelation are preferred
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/CfsSubsetEval.html

	MRMR_MI_BASED_SUBSET,
	//Minimum-redundancy-maximum-relevance using mi
	//LINK: http://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_.28mRMR.29_feature_selection

	FCBF,
	//Feature selection method based on correlation measureand relevance&redundancy analysis. 
	//link: http://weka.sourceforge.net/doc.packages/fastCorrBasedFS/weka/attributeSelection/FCBFSearch.html
	
	//wrappers ******

	SVMRFE,
	//Attributes are ranked by the square of the weight assigned by the SVM. 
	//Attribute selection for multiclass problems is handled by ranking attributes for each class seperately using a one-vs-all method and then "dealing" from the top of each pile to give a final ranking.
	//link: http://weka.sourceforge.net/doc.packages/SVMAttributeEval/weka/attributeSelection/SVMAttributeEval.html
	
	FORWARD_SELECTION_WRAPPER,
	//wrapper foward search technique

	BACKWARD_SELECTION_WRAPPER,
	//wrapper BACKWARD search technique
	
	//our approach****************
	
	HIGH_PRE_EXPECT_APP,
	//our model to optimize the expectation and to provide high precision
	
	HIGH_PRE_LOGLIK_APP,
	//our model to optimize the log likelihood and to provide high precision
	
	HIGH_REC_EXPECT_APP,
	//our model to optimize the expectation and to provide high recall
	
	HIGH_REC_LOG_APP,
	//our model to optimize the log likelihood and to provide high recall

	EXPECTATION_GENERAL,
	//n out of k expectation 
	
	LOGLIKELIHOOD_GENERAL
	// n out of k loglikelihood
	
}
