package tests;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import junit.framework.Assert;

import org.junit.Test;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;

import com.google.common.collect.Sets;

import evaluation.CrossValidationOutput;
import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.ClassificationExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;

public class BackGroundTest {


	//test read a arff file
	@Test
	public void testReadProblems() {

		String filePath = "./TestDataSets/lsdata1.arff";
		ClassificationProblem cp;
		try {
			cp = new ClassificationProblem(filePath);

			Assert.assertNotNull("can not read the data",cp.getData());
			Assert.assertTrue("arff relation name wrong",cp.getName().equals("LinearSeparableYequalX"));
			Assert.assertTrue("path wrong",cp.getFilePath().equals(".\\TestDataSets\\lsdata1.arff"));
			Assert.assertTrue("don't read all examples",cp.getNumExamples() == 300);
			Assert.assertTrue("didn't read all atributes",cp.getNumAttributes() == 3);
			Assert.assertTrue("class index wrong",cp.getData().classIndex() == 2);

		} catch (IOException e) {
			fail("Problem to read the arrff file: " + e.getMessage());
		}
	}

	//test classifier in artificial linear sparable data
	@Test
	public void testClassifiersInArtificialData() {

		try{

			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/lsdata1.arff");
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(cp);

			AbstractClassifier nb = new NaiveBayesClassifier();
			AbstractClassifier lr = new LogisticRegressionClassifier();
			AbstractClassifier svm = new SVMLinearClassifier();


			nb.buildClassifier(cp.getData());
			ev.evaluateModel(nb,cp.getData());
			double error_rate_nb = ev.errorRate();


			lr.buildClassifier(cp.getData());
			ev.evaluateModel(lr,cp.getData());
			double error_rate_lr = ev.errorRate();


			svm.buildClassifier(cp.getData());
			ev.evaluateModel(svm,cp.getData());
			double error_rate_svm = ev.errorRate();


			Assert.assertTrue("Naive byes' erro rate for linear separable data have less than 1%", error_rate_nb < 0.01 );
			Assert.assertTrue("Logistic Regression's erro rate for linear separable data have to be zer0",error_rate_lr == 0 );
			Assert.assertTrue("Linear SVM's erro rate for linear separable data have to be zero",error_rate_svm == 0 );


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//test classifier and cross validation in a data base with know performance 
	//TODO: No refrence to NB and LR. There is reference just to Linear SVM (near the reference 0.04 of distance) 
	@Test
	public void testClassifierInUCIData() {
		try{
			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/heart-statlog.arff");

			AbstractClassifier nb = new NaiveBayesClassifier();
			AbstractClassifier lr = new LogisticRegressionClassifier();
			AbstractClassifier svm = new SVMLinearClassifier();

			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);

			eval.crossValidateModelTuneInAccuraccy(nb, null,cp, 10, 10, null);
			//Assert.assertTrue("Naive Bayes performance not expected", Math.abs(eval.accuracy()-0.75) < 0.05);

			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",utils.Util.generateModelsStringSettings("-C", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			paramLR.put("-B",utils.Util.generateModelsStringSettings("-B", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			eval.crossValidateModelTuneInAccuraccy(lr, null,cp, 10, 10, paramLR);
			Assert.assertTrue("Logistic Regression performance not expected", Math.abs(eval.accuracy()-0.85) < 0.05);


			HashMap<String,Set<String>> paramSVM = new HashMap<String,Set<String>>();
			paramSVM.put("-C",utils.Util.generateModelsStringSettings("-C", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			paramSVM.put("-B",utils.Util.generateModelsStringSettings("-B", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			eval.crossValidateModelTuneInAccuraccy(svm, null,cp, 10, 10, paramSVM);
			Assert.assertTrue("SVM performance not expected", Math.abs(eval.accuracy() - 0.86) < 0.05);


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//test run in jarfiles and files
	@Test
	public void testExperimentExecutor() {
		IExperimentCommand cmd = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport("test", "test"));
				return exp;
			}
		};
		String dataPath = "./data/iris.data.arff";
		ClassificationExperimentReport exprep = (ClassificationExperimentReport)ExperimentExecutor.getInstance().executeCommandInFile(cmd, dataPath);
		Assert.assertTrue("", exprep.getProblemName().equals("test"));

		IExperimentCommand cmd2 = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath2 = "./data";
		List<AbstractExperimentReport> exprep2 =  ExperimentExecutor.getInstance().executeCommandInFiles(cmd2, dataPath2);
		Assert.assertNotNull("Returning null in execute experiemnts in a diretory", exprep2);
		Assert.assertTrue("Retrunning zero items in execute experiemnts in a diretory", exprep2.size() > 0);

		IExperimentCommand cmd3 = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath3 = "./data/datasets-UCI.jar";
		List<AbstractExperimentReport> exprep3 = ExperimentExecutor.getInstance().executeCommandInJAR(cmd3, dataPath3);
		Assert.assertNotNull("Execute Experiment in jar data set: return null", exprep3);
		Assert.assertTrue("Execute Experiment in jar data set have to get 36 reports(==36 problems)" + exprep3.size(), exprep3.size() == 36);

	}

	//test cross validation
	//TODO: check if the crossvalidation have to give the best results
	@Test
	public void testCrossValidation() {
		try{
			ClassificationProblem p = new ClassificationProblem("./data/iris.data.arff");
			AbstractClassifier lr = new LogisticRegressionClassifier();
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(p);

			//cross validation with default parameters
			CrossValidationOutput cvo = ev.crossValidateModelTuneInAccuraccy(lr, null,p, 10, 10, null);
			double accuracy = cvo.metricMean(EClassificationMetric.ACCURACY);
			double precision = cvo.metricMean(EClassificationMetric.PRECISION);
			double recall = cvo.metricMean(EClassificationMetric.RECALL);
			double fmeasure = cvo.metricMean(EClassificationMetric.FSCORE);

			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0"));
			paramLR.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0"));

			//return the classifier already tuned with cross validation with parameters
			CrossValidationOutput cvo2 = ev.crossValidateModelTuneInAccuraccy(lr, null,p, 10, 10, paramLR);
			double accuracy_aftercv = cvo2.metricMean(EClassificationMetric.ACCURACY);
			double precision_aftercv = cvo2.metricMean(EClassificationMetric.PRECISION);
			double recall_aftercv = cvo2.metricMean(EClassificationMetric.RECALL);
			double fmeasure_aftercv = cvo2.metricMean(EClassificationMetric.FSCORE);


			Assert.assertTrue("The cross validated classifier have to be better in accuracy", accuracy_aftercv >=  accuracy);
			Assert.assertTrue("The cross validated classifier have to be better in precision", precision_aftercv >=  precision);
			Assert.assertTrue("The cross validator classifier have to be better in recall", recall_aftercv >=  recall);
			Assert.assertTrue("The cross validator classifier have to be better in fmeasure", fmeasure_aftercv >=  fmeasure);


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//full weka evaluation metrics computation test and wekawrapperevaluation metric computation test
	@Test
	public void testMetrics(){
		//initializations
		String filePath = "./TestDataSets/breast-cancer.arff";
		ClassificationProblem cp;
		try {
			//format to compute at least 4 digital fractions
			final NumberFormat nf = NumberFormat.getInstance();
			nf.setMinimumFractionDigits(4);
			nf.setMaximumFractionDigits(4);
			nf.setGroupingUsed(false);
			
			//read a data set
			cp = new ClassificationProblem(filePath);

			//classifier and evaluators
			AbstractClassifier svm = new SVMLinearClassifier();
			Evaluation  eval = new Evaluation(cp.getData());
			WekaEvaluationWrapper wrapper = new WekaEvaluationWrapper(cp);
			
			//train and evaluate
			svm.buildClassifier(cp.getData());
			eval.evaluateModel(svm,cp.getData());
			wrapper.evaluateModel(svm, cp.getData());
			
			
			//weka evaluation accurracy
			double weka_accuracy = 1 - eval.errorRate();
			
			//my computation accuracy
			double accuracyNumerator = 0;
			double accuracyDenominator = 0;
			
			//wrapper evaluation variables
			double precisionMean = 0;
			double recallMean = 0;
			double FmeasureMean = 0;
			
			//compute metrics
			for(int i = 0 ; i < cp.getData().classAttribute().numValues(); i ++){
				double TP = 0;
				double TN = 0;
				double FP = 0;
				double FN = 0;
				for (Instance datum : cp.getData()) {					
					double actual = datum.classValue();
					double pred = svm.classifyInstance(datum);

					if(i == actual){
						if(pred == actual){
							//tp++
							TP++;
						}else{
							//fn++;
							FN++;
						}
					}else{
						if(pred == actual){
							//tn++
							TN++;
						}else{
							//fp++;
							FP++;
						}
					}
				}
				
				//my computation: precision, recall fmeasure for a given class i
				double precision = (TP/(TP+FP));
				double recall = (TP/(TP+FN));
				double fmeasure = (2* precision * recall) / (precision + recall);
				
				//computation of mean and general accuracy
				accuracyNumerator += TP+TN;
				accuracyDenominator += TP+TN+FP+FN;
				 precisionMean += precision;
				 recallMean += recall;
				 FmeasureMean += fmeasure;
				
				//weka computation
				double weka_precision = eval.precision(i);
				double weka_recall = eval.recall(i);
				double weka_fmeasure = eval.fMeasure(i);
				
				//check if weka results and mine are the same
				Assert.assertEquals("Weka Precision Wrong", nf.format(precision), nf.format(weka_precision));
				Assert.assertEquals("Weka recall Wrong", nf.format(recall), nf.format(weka_recall));
				Assert.assertEquals("Weka fscore Wrong", nf.format(fmeasure), nf.format(weka_fmeasure));
				
			}
			
			//compute general metrics
			double accuraccy = accuracyNumerator / accuracyDenominator;
			precisionMean = precisionMean / cp.getData().classAttribute().numValues();
			recallMean = recallMean / cp.getData().classAttribute().numValues();
			FmeasureMean = FmeasureMean / cp.getData().classAttribute().numValues();
			
			//check wraper and general metrics
			Assert.assertEquals("Weka Accuracy Wrong", nf.format(accuraccy), nf.format(weka_accuracy));
			Assert.assertEquals("Weka Accuracy in wrapper Wrong", nf.format(accuraccy), nf.format(wrapper.accuracy()));
			Assert.assertEquals("Weka precision in wrapper Wrong", nf.format(precisionMean), nf.format(wrapper.precision()));
			Assert.assertEquals("Weka recall in wrapper Wrong", nf.format(recallMean), nf.format(wrapper.recall()));
			Assert.assertEquals("Weka fmeasure in wrapper Wrong", nf.format(FmeasureMean), nf.format(wrapper.fMeasure()));

		} catch (Exception e) {
			fail("Problem to read the arrff file: " + e.getMessage());
		}
	}

}
