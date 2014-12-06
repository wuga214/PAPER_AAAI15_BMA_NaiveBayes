package classifiers;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

import com.google.common.collect.Sets;

import evaluation.WekaEvaluationWrapper;
import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SVMLinearClassifier extends LibLINEAR {


	//constructor
	public SVMLinearClassifier(){
		this.setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
	}

	@Override
	public void buildClassifier(Instances data)
			throws Exception {

		//set the liblinear to use L2-loss support vector machines (dual)
		this.setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));

		//build the classify (= train)
		super.buildClassifier(data);

	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		//set the liblinear to use L2-loss support vector machines (dual)
		this.setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
		//classify the new instance
		return super.classifyInstance(arg0);
	}



	@Override
	public void setOptions(String[] options) throws Exception {

		//work around to avoid the api print trash in the console
		PrintStream console = System.out;
		System.setOut(new PrintStream(new OutputStream() {
			@Override public void write(int b) throws IOException {}
		}));

		super.setOptions(options);

		//set the liblinear to use l2 regularized logistic regression
		this.setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));

		//work around to avoid the api print trash in the console
		System.setOut(console);
	}


	//test the clasifier implmentation
	public static void main(String[] args) {

		try {

			AbstractClassifier classifier = new  SVMLinearClassifier();
			ClassificationProblem cp = new ClassificationProblem("./data/UCI/anneal.arff");
			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);	
			
		//	long startTime = System.nanoTime();
		//	classifier.buildClassifier(cp.getData());
			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",Sets.newHashSet("-C 0.01","-C 0.1","-C 1","-C 10","-C 100","-C 1000","-C 10000"));
			paramLR.put("-B",Sets.newHashSet("-B 1","-B 10"));
		//	cp.setData(DALinearPlot.deleteAttribute(cp.getData(),cp.getData().numAttributes()-22));
			eval.CrossValidation(classifier, null,cp, 10, 10, paramLR,null);
//			classifier.distributionForInstance(cp.getData().instance(0));
	//		long endTime = System.nanoTime();
	//		long duration = endTime - startTime;
			System.out.println(eval.accuracy());
		//	System.out.println(duration);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}







}
