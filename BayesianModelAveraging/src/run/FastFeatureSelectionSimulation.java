package run;

import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang3.time.StopWatch;

import ExperimentCommands.FastFeatureSelectionCommand;
import classifiers.ELinearClassifier;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;

public class FastFeatureSelectionSimulation {
	public static void main(String[] args) {

		StopWatch stp = new StopWatch();stp.start();

		//parameters of simulation***********
		
		//Paths
		String dataSetFolderpath = "./binary_data";
		String graphOutPutFolderPath = "./results/graphs";
		String csvResultsPath = "./results/featureSelection.csv";
		String xmlResultPath = "./results/featureSelection.xml";
		
		//limit of features to select eg 50 -> run from 0 to 50 features
		int KFeatures = 50; 
		
		//number of folds in cros validation. However remember to change the costant 2.62 in the plots or in the experiment report class
		int folds = 10;
		
		//boolean to say if is to tune the model in each metric or just in accuracy.
		boolean tuneJustInAccuracy = false;
		
		//clasifiers
		List<ELinearClassifier> classifiers = Lists.newArrayList(
				//ELinearClassifier.LOGISTIC_REGRESSION
				//,ELinearClassifier.SVM_LINEAR
				 //,ELinearClassifier.NAIVE_BAYES
				ELinearClassifier.C_BMA
				);

		// feature selection algorithms
		List<EFeatureSelectionAlgorithm> selectionAlgs = Lists.newArrayList(
				EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK
				,EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK
				//,EFeatureSelectionAlgorithm.GAINRATIO_RANK
				//EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK
				//,EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK
				//,EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET
				//EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.FCBF
				//,EFeatureSelectionAlgorithm.RELIFF
				//,EFeatureSelectionAlgorithm.SVMRFE
				//,EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.HIGH_PRE_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_PRE_LOGLIK_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_LOG_APP
				//EFeatureSelectionAlgorithm.EXPECTATION_GENERAL
				//,EFeatureSelectionAlgorithm.LOGLIKELIHOOD_GENERAL
				);
		

		
		
		//Fast simulation
		IExperimentCommand cmd = new FastFeatureSelectionCommand(classifiers ,selectionAlgs, graphOutPutFolderPath, KFeatures, folds,tuneJustInAccuracy);
		
		//execute the simulation
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, dataSetFolderpath);

		utils.Util.writeToXml(result,xmlResultPath);
		
		//save results in csv
		AbstractExperimentReport.saveAll(result, csvResultsPath);

		//time elapsed computation
		stp.stop();
		
		System.out.println("Total simulation time: (Hours:Minutes:Second.Milisecond): " + stp.toString());
	}
}
