package run;

import java.util.List;

import org.apache.commons.lang3.time.StopWatch;

import ExperimentCommands.FeatureSelectionCommand;
import classifiers.ELinearClassifier;

import com.google.common.collect.Lists;

import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;

public class FeatureSelectionSimulation {
	public static void main(String[] args) {

		StopWatch stp = new StopWatch();
		stp.start();
		
		//parameters of simulation
		String dataSetFolderpath = "./data";
		String graphOutPutFolderPath = "./results/graphs";
		String csvResultsPath = "./results/featureSelection.csv";
		int folds = 10;
		int kFeatures = 30;
		//boolean to say if is to tune the model in each metric or just in accuracy.
		boolean tuneJustInAccuracy = false;

		//classifiers
		List<ELinearClassifier> classifiers = Lists.newArrayList(
				ELinearClassifier.LOGISTIC_REGRESSION
				,ELinearClassifier.SVM_LINEAR
				,ELinearClassifier.NAIVE_BAYES
				);

		//feature selection algorithms
		List<EFeatureSelectionAlgorithm> selectionAlgs = Lists.newArrayList(
				//EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK
				//,EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK
				//,EFeatureSelectionAlgorithm.GAINRATIO_RANK
				EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK
				//,EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK
				//,EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.FCBF
				//,EFeatureSelectionAlgorithm.RELIFF
				//,EFeatureSelectionAlgorithm.SVMRFE
				//,EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.HIGH_PRE_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_PRE_LOGLIK_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_LOG_APP
				);

		//Normal feature selection command
		IExperimentCommand cmd = new FeatureSelectionCommand(classifiers ,selectionAlgs, graphOutPutFolderPath, kFeatures, folds, tuneJustInAccuracy);
		
		//get the results
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, dataSetFolderpath);

		//save results in csv
		AbstractExperimentReport.saveAll(result, csvResultsPath);

		//time elapsed computation
		stp.stop();
		System.out.println("elapsed time: " + stp.toString());
	}
}
