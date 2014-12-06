package ExperimentCommands;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.lang3.time.StopWatch;

import problems.ClassificationProblem;
import utils.Util;
import weka.attributeSelection.ASEvaluation;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import classifiers.ClassifierFactory;
import classifiers.ELinearClassifier;

import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import evaluation.CrossValidationOutput;
import evaluation.EClassificationMetric;
import evaluation.FoldResult;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.FeatureSelectionExperimentReport;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;

public class FastFeatureSelectionCommand implements IExperimentCommand{

	private String GraphsPath;
	private  List<EFeatureSelectionAlgorithm> selectionAlgs = null;
	private List<ELinearClassifier> classifiers = null;
	private Logger log = Util.getFileLogger(FeatureSelectionCommand.class.getName(), "./results/logs/log.txt");
	private int kMaxFeatures = Integer.MAX_VALUE;
	private boolean tuneJustInAccuracy = true;
	private StopWatch timer;
	private int folds = 10;

	public FastFeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath){

		this(classifiers,FetSelectionAlgs,graphPath, Integer.MAX_VALUE, 10, true);

	}
	public FastFeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath, int kMaxFeatures, int folds, boolean tuneJustInAccuracy){

		this.GraphsPath = graphPath;
		this.selectionAlgs = FetSelectionAlgs;
		this.classifiers = classifiers;
		this.kMaxFeatures = kMaxFeatures;
		this.timer = new StopWatch(); this.timer.start();
		this.folds = 10;
		this.tuneJustInAccuracy = tuneJustInAccuracy;
	}

	@Override
	public List<AbstractExperimentReport> execute(ClassificationProblem cp)  {

		System.out.println("> problem: " + cp.getName());

		List<AbstractExperimentReport> result = new ArrayList<AbstractExperimentReport>();
		WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);


		for (ELinearClassifier ecl :  this.classifiers){

			System.out.println(">> classifier: " + ecl.name());
			long classifierTime = timer.getTime();

			//get the classifier and the parameters
			AbstractClassifier cl = ClassifierFactory.getInstance().createClassifier(ecl);
			Map<String,Set<String>> param =  ClassifierFactory.getInstance().getDefaultClassifiersParameters(ecl);


			// calculate the max number of features to the simulation with this data set and instatiate reports
			int maxNumFeatures = kMaxFeatures < cp.getNumAttributes() - 1 ? kMaxFeatures : cp.getNumAttributes() - 1;
			FeatureSelectionExperimentReport exp_acc = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.ACCURACY.name());
			FeatureSelectionExperimentReport exp_pre = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.PRECISION.name());
			FeatureSelectionExperimentReport exp_rec = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds,  EClassificationMetric.RECALL.name());
			FeatureSelectionExperimentReport exp_fm = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(),  maxNumFeatures, folds, EClassificationMetric.FSCORE.name());


			//run all selected feature selection algorithm
			for (EFeatureSelectionAlgorithm alg : this.selectionAlgs) {

				System.out.println(">>> feature selection alg: " + alg.name());
				long featureSelectionAlgTime = timer.getTime();

				try {
					//variable to speedup the simulation [foldNumber] -> previous selected[]
					int[][] fold2PreviousSelectedFeature = new int[folds][];
					ASEvaluation[] fold2evaluatior = new ASEvaluation[folds];

					//randomize the samples
					Random rand = new Random(System.currentTimeMillis()); 
					Instances randData = new Instances(cp.getData());   
					randData.randomize(rand);

					//run for many number of features select
					for (int n_features = 1; n_features <= maxNumFeatures; n_features++) {

						//debug timer
						//long featureTime = timer.getTime();

						CrossValidationOutput outlr = null;
						if(this.tuneJustInAccuracy){
							//cross validate - the feature algorithm is created inside to speed up the simulation
							outlr = eval.fastCrossValidationTuneInAccuracy(cl, randData, n_features, alg, folds, param, fold2PreviousSelectedFeature, fold2evaluatior);
						}else{
							//cross validate - the feature algorithm is created inside to speed up the simulation
							outlr = eval.fastCrossValidation(cl, randData, n_features, alg, folds, param, fold2PreviousSelectedFeature, fold2evaluatior,Lists.newArrayList(EClassificationMetric.values()));
						}

						//restrieve the previous selected features by fold
						for (int foldNumber = 0; foldNumber < outlr.getFolds(); foldNumber++) {
							FoldResult fr = outlr.getFoldResult(foldNumber);
							if(fr != null){
								fold2PreviousSelectedFeature[foldNumber] = fr.getFeatureSelected();
								fold2evaluatior[foldNumber] = fr.getEvaluatorBuilt();
							}
						}

						//collect cross validation measurement
						exp_acc.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.ACCURACY));
						exp_pre.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.PRECISION));
						exp_rec.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.RECALL));
						exp_fm.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.FSCORE));

						exp_acc.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.ACCURACY));
						exp_pre.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.PRECISION));
						exp_rec.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.RECALL));
						exp_fm.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.FSCORE));

						//debug
						//System.out.println("elapsed time to select the " + n_features + " (th) feature: " + DurationFormatUtils.formatDuration(timer.getTime() - featureTime, "HH:mm:ss.S"));

					}

					System.out.println(">>> "+ alg.name() + " done! Elapsed Time( HH:mm:ss.S): " + 
							DurationFormatUtils.formatDuration(timer.getTime() - featureSelectionAlgTime, "HH:mm:ss.S") );

				} catch (Exception e) {
					String msg = Joiner.on(" ").skipNulls().join("problem in:",cp.getName(),ecl.name(),alg.name());
					log.log(Level.WARNING, msg , e);
					System.out.println(msg);
				}

			}


			//plot graphs
			this.savePartialResuls(result, exp_acc, exp_pre, exp_rec, exp_fm);

			System.out.println(">> " + ecl.name() + " done! Elapsed Time(HH:mm:ss.S): " 
					+ DurationFormatUtils.formatDuration(timer.getTime() - classifierTime, "HH:mm:ss.S"));
		}

		System.out.println(">Problem " + cp.getName() + " done! Actual Simulation time: " + timer.toString() );

		return result;
	}


	private void savePartialResuls( List<AbstractExperimentReport> results, AbstractExperimentReport... exps){
		for (AbstractExperimentReport exp : exps) {
			exp.plot(GraphsPath);
			results.add(exp);
		}
	}


}
