package classifiers;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;

import weka.classifiers.AbstractClassifier;

public class ClassifierFactory {

	//singleton impl
	private static ClassifierFactory instance;
	public synchronized static ClassifierFactory getInstance(){
		if(instance == null)
			instance = new ClassifierFactory();

		return instance;
	}
	public ClassifierFactory(){}

	//create classifier to be used in the simulations 
	public AbstractClassifier createClassifier(ELinearClassifier EType) {
		AbstractClassifier ret = null;

		switch (EType) {
		case LOGISTIC_REGRESSION :
			ret = new LogisticRegressionClassifier();
			break;

		case SVM_LINEAR :
			ret = new SVMLinearClassifier();
			break;

		case NAIVE_BAYES :
			ret = new NaiveBayesClassifier();
			break;
			
		case C_NAIVE_BAYES :
			ret = new GaussianNaiveBayes();
			break;
			
		case C_BMA :
			ret = new BMAOfNBWithGaussian();
			break;
		}

		return ret;
	}
	
	public Map<String,Set<String>> getDefaultClassifiersParameters(ELinearClassifier etype){
		Map<String,Set<String>> param = new HashMap<String,Set<String>>();

		switch (etype) {
		case LOGISTIC_REGRESSION :
			param.put("-C",Sets.newHashSet("-C 0.0001","-C 0.001","-C 0.01","-C 0.1","-C 1","-C 10","-C 100","-C 1000","-C 10000","-C 0.05","-C 0.5","-C 5","-C 50","-C 500"));
			param.put("-B",Sets.newHashSet("-B 1"));
			break;

		case SVM_LINEAR :
			param.put("-C",Sets.newHashSet("-C 0.0001","-C 0.001","-C 0.01","-C 0.1","-C 1","-C 10","-C 100","-C 1000","-C 10000","-C 0.05","-C 0.5","-C 5","-C 50","-C 500"));
			param.put("-B",Sets.newHashSet("-B 1"));
			break;
			
		case C_BMA :
			param.put("-Q",Sets.newHashSet(" -Q"));
			param.put("-H",Sets.newHashSet("-H 0.5","-H 0.6","-H 0.7","-H 0.8","-H 0.9","-H 1","-H 1.1","-H 1.2","-H 1.3","-H 1.4","-H 1.5"));
			break;

		case NAIVE_BAYES :
			param = null;
			break;
			
		case C_NAIVE_BAYES :
			param = null;
			break;
		}

		return param;

	}
	
}
