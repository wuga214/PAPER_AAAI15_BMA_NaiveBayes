package tests;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import junit.framework.Assert;

import org.junit.Test;

import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;
import evaluation.CrossValidationOutput;
import evaluation.FoldResult;
import evaluation.WekaEvaluationWrapper;

public class WekaAPITests {

	//test weka evaluation class. it will fail because the weka evaluation class dont reset in each call of evaluatemodel
	//because that i coded wekaevaluationwrapper
	@Test
	public void testWEKAEvaluation() {
		try{
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			AbstractClassifier classifier = new NaiveBayesClassifier();

			Evaluation eval = new Evaluation(cp.getData());

			//eval 1
			classifier.buildClassifier(cp.getData());
			eval.evaluateModel(classifier, cp.getData());
			double errorate1 = eval.errorRate();
			double precision1 = eval.precision(0);
			double recall1 = eval.recall(0);
			double correct1 = eval.correct();

			//eval 2
			classifier.buildClassifier(cp.getData());
			eval.evaluateModel(classifier, cp.getData());
			double errorate2 = eval.errorRate();
			double precision2 = eval.precision(0);
			double recall2 = eval.recall(0);
			double correct2 = eval.correct();

			Assert.assertEquals("Error rate have to be the same for two execution of the evaluation with the same object and settings, but it isn't",errorate1, errorate2);
			Assert.assertEquals("Precision have to be the same for two execution of the evaluation with the same object and settings, but it isn't",precision1, precision2);
			Assert.assertEquals("Recall rate have to be the same for two execution of the evaluation with the same object and settings,but it isn't",recall1, recall2);
			Assert.assertEquals("Number of correct have to be the same for two execution of the evaluation with the same object and settings, but it isn't",correct1, correct2);

			//new eval
			Evaluation eval2 = new Evaluation(cp.getData());
			classifier.buildClassifier(cp.getData());
			eval2.evaluateModel(classifier, cp.getData());
			double errorate3 = eval2.errorRate();
			double precision3 = eval2.precision(0);
			double recall3 = eval2.recall(0);
			double correct3 = eval2.correct();

			Assert.assertEquals("Error rate have to be the same for two eval objects with the same  settings",errorate1, errorate3);
			Assert.assertEquals("Precision have to be the same for two eval objects with the same  settings",precision1, precision3);
			Assert.assertEquals("Recall rate have to be the same for two eval objects with the same  settings",recall1, recall3);
			Assert.assertEquals("Recall rate have to be the same for two eval objects with the same  settings",correct2, correct3);


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//test if subsequent calling to build classifier make the classifier training continously 
	//or in each call it is reseted automatically. It is reseted...
	@Test
	public void testWEKABuildClassfier(){

		try {
			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/heart-statlog.arff");

			AbstractClassifier nb = new NaiveBayesClassifier();
			AbstractClassifier lr = new LogisticRegressionClassifier();
			AbstractClassifier svm = new SVMLinearClassifier();

			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);

			//nb
			nb.buildClassifier(cp.getData());
			eval.evaluateModel(nb, cp.getData());
			double accuracy1 = eval.accuracy();

			nb.buildClassifier(cp.getData());
			eval.evaluateModel(nb, cp.getData());
			double accuracy2 = eval.accuracy();
			Assert.assertTrue("The accuracy or other metric have to be the same after subsequent naivebayes.buildclassifier", accuracy1 == accuracy2);

			//lr
			lr.buildClassifier(cp.getData());
			eval.evaluateModel(lr, cp.getData());
			accuracy1 = eval.accuracy();

			lr.buildClassifier(cp.getData());
			eval.evaluateModel(lr, cp.getData());
			accuracy2 = eval.accuracy();
			Assert.assertTrue("The accuracy or other metric have to be the same after subsequent logisticRegression.buildclassifier", accuracy1 == accuracy2);

			//svm
			svm.buildClassifier(cp.getData());
			eval.evaluateModel(svm, cp.getData());
			accuracy1 = eval.accuracy();

			svm.buildClassifier(cp.getData());
			eval.evaluateModel(svm, cp.getData());
			accuracy2 = eval.accuracy();
			Assert.assertTrue("The accuracy or other metric have to be the same after subsequent svm.buildclassifier", accuracy1 == accuracy2);



		} catch (Exception e) {
			fail("can not read the file or problem in evaluation");
			e.printStackTrace();
		}
	}

	@Test
	public void testFilters(){
		try {
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			Instances data = cp.getData();
			Remove rm = new Remove();
			rm.setAttributeIndicesArray(new int[]{1,4});
			rm.setInputFormat(data);
			Instances res = Filter.useFilter(data, rm);

			System.out.println(res == data);


		} catch (IOException e) {
			fail("cant read the data set");
			e.printStackTrace();
		} catch (Exception e) {

			e.printStackTrace();
		}
	}

}
