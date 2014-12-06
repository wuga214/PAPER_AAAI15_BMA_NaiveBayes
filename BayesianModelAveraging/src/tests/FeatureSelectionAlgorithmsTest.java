package tests;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

import junit.framework.Assert;

import org.junit.Test;

import problems.ClassificationProblem;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import JavaMI.Entropy;
import JavaMI.MutualInformation;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;
import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelector;
import featureSelection.MyGreedySearch;
import featureSelection.Evaluators.MRMRFeatureSelection;


public class FeatureSelectionAlgorithmsTest {

	@Test
	public void TestNewAttributeSelection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		//this test work for all evaluator unless cfs but the error is the weka code because it adds more feature than requested
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			ASEvaluation ev1 = new MRMRFeatureSelection();
			ev1.buildEvaluator(cp.getData());
			
			ASEvaluation ev2 = new MRMRFeatureSelection();
			ev2.buildEvaluator(cp.getData());

			MyGreedySearch s = new MyGreedySearch();
			s.setNumToSelect(3);

			featureSelection.FeatureSelector f1 = new featureSelection.FeatureSelector();
			f1.setEvaluator(ev1);
			f1.setPrebuilt(true);
			f1.setSearch(s);

			AttributeSelection f2 = new AttributeSelection();
			f2.setEvaluator(ev2);
			f2.setSearch(s);
			
			
			
			int[] f1idx = f1.selectAttributes(cp.getData());
			
			f2.SelectAttributes(cp.getData());
			int[] f2idx = f2.selectedAttributes();
		
			
			Assert.assertTrue("The Results were different", Arrays.equals(f1idx, f2idx));
			
			s.setNumToSelect(5);
			 f1idx = f1.selectAttributes(cp.getData());
			
			f2.SelectAttributes(cp.getData());
			f2idx = f2.selectedAttributes();
			
			
			Assert.assertTrue("The Results were different", Arrays.equals(f1idx, f2idx));

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}
	
	@Test
	public void TestGreedyAlgorithmModification(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			
			ASEvaluation ev1 = new MRMRFeatureSelection();
			ASEvaluation ev2 = new MRMRFeatureSelection();

			GreedyStepwise s1 = new GreedyStepwise();
			//s1.setSearchBackwards(true);
			s1.setGenerateRanking(true);
			s1.setNumToSelect(1);

			MyGreedySearch s2 = new MyGreedySearch();
			//s2.setSearchBackwards(true);
			s2.setNumToSelect(1);

			AttributeSelection f1 = new AttributeSelection();
			f1.setEvaluator(ev1);
			f1.setSearch(s1);

			AttributeSelection f2 = new AttributeSelection();
			f2.setEvaluator(ev2);
			f2.setSearch(s2);
			
			
			f1.SelectAttributes(cp.getData());
			int[] f1idx = f1.selectedAttributes();
			
			f2.SelectAttributes(cp.getData());
			int[] f2idx = f2.selectedAttributes();
		
			
			Assert.assertTrue("The Results were different", Arrays.equals(f1idx, f2idx));

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testJavaMIAPI() {

		DecimalFormat  formatter = new DecimalFormat ("#0.00"); 
		double[] X = new double[]{1,1,2,2,3,3,3,4,4,4,5,5,5,5,6,6};//l
		double[] Y = new double[]{1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3};//v

		//MI(X,Y) = MI(Y,X) - log2 acording to the documentation
		double mi = MutualInformation.calculateMutualInformation(Y, X);
		//System.out.println(mi);
		assertTrue("MI(Y|X) = 1.25, therefore the result is wrong", Double.valueOf(formatter.format(mi)) == 1.25);

		//H(X|Y) - log2 acording to the documentation
		double ce = Entropy.calculateConditionalEntropy(X, Y);
		//System.out.println(ce);
		assertTrue("H(X|Y) = 1.28, therefore the result is wrong", Double.valueOf(formatter.format(ce)) == 1.28);

		//H(X) - log2
		double e = Entropy.calculateEntropy(X);
		//System.out.println(e);
		assertTrue("H(X) = 2.53, therefore the result is wrong", Double.valueOf(formatter.format(e)) == 2.53);

		//MI(Y,X) = MI(X,Y) = H(X) - H(X|Y)
		double mi2 = Double.valueOf(formatter.format(e - ce)) ;
		assertTrue("MI(Y,X) = MI(X,Y) = H(X) - H(X|Y), therefore the result is wrong", Double.valueOf(formatter.format(mi)) == mi2);

	}

	@Test
	public void testInformationGainFeatureSelection() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}



		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testConditionalEntropy() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testGainRationRank() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.GAINRATIO_RANK, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.GAINRATIO_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void SymmetricalUncertRank() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testCorrelationBasedRankFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testCorrelationBasedSubsetFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testMRMRFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET, parameter);

			//exeute
			System.out.println(EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET.name());
			int[] idxs = filter.selectAttributes(cp.getData());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	@Test
	public void testForwardFeatureSelectionAlgorithm(){

		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(2, new NaiveBayesClassifier(), cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}


		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	@Test
	public void testBackwardFeatureSelectionAlgorithm(){

		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(2, new NaiveBayesClassifier(), cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	@Test
	public void testRelif(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			AttributeSelection filter = new AttributeSelection();
			//reliff evaluator
			ASEvaluation evaluator = new ReliefFAttributeEval();
			//rank by evaluator values
			ASSearch search  = new Ranker();
			((Ranker) search).setNumToSelect(5);


			filter.setEvaluator(evaluator);
			filter.setSearch(search);
			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.RELIFF.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	@Test
	public void testFCBF(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, new NaiveBayesClassifier(), cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.FCBF, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.FCBF.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	@Test
	public void testSVMRFE(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, new SVMLinearClassifier(), cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.SVMRFE, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.SVMRFE.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	@Test
	public void testHighPrecEpctApp(){
		String dataset = "./binary_data/vote.arff";
		dataset = "./data/autos.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.HIGH_PRE_EXPECT_APP, parameter);


			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.HIGH_PRE_EXPECT_APP.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}
	@Test
	public void testHighPrecLogLApp(){
		String dataset = "./binary_data/vote.arff";
		dataset = "./data/autos.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.HIGH_PRE_LOGLIK_APP, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.HIGH_PRE_LOGLIK_APP.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}
	@Test
	public void testHighRecEpctApp(){
		String dataset = "./binary_data/vote.arff";
		dataset = "./binary_data/newsgroupstest.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.HIGH_REC_EXPECT_APP, parameter);

			//exeute
			
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.HIGH_REC_EXPECT_APP.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}
	@Test
	public void testHighRecLogLApp(){
		String dataset = "./binary_data/vote.arff";
		dataset = "./binary_data/newsgroupstest.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			FeatureSelector filter = FeatureSelector.create(EFeatureSelectionAlgorithm.HIGH_REC_LOG_APP, parameter);

			//exeute
			int[] idxs = filter.selectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.HIGH_REC_LOG_APP.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}


}
