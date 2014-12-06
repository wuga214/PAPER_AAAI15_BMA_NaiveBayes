package classifiers;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class NaiveBayesClassifier extends NaiveBayes {

	//constructor 
	public NaiveBayesClassifier(){
		super();
	}

	//method to train a naive bayes classifier
	@Override
	public void buildClassifier(Instances data)throws Exception {
		super.buildClassifier(data);

	}

	//method to classify a new intance using the naive bayes classifyer
	@Override
	public double classifyInstance(Instance newInstance) throws Exception {
		return super.classifyInstance(newInstance);

	}
	@Override


	public void setOptions(String[] options) throws Exception {
		//work around to avoid the api print trash in the console
		PrintStream console = System.out;
		System.setOut(new PrintStream(new OutputStream() {
			@Override public void write(int b) throws IOException {}
		}));

		super.setOptions(options);

		//work around to avoid the api print trash in the console
		System.setOut(console);
	}



	//test classifier
	public static void main(String[] args) {

		try {
			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/breast-cancer.arff");
			AbstractClassifier classifier = new NaiveBayesClassifier();
			classifier.buildClassifier(cp.getData());
			for (Instance instance : cp.getData()) {
				System.out.print(classifier.classifyInstance(instance) + ", ");
			}
			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
