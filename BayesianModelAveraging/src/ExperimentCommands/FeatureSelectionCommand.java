package ExperimentCommands;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.lang3.time.StopWatch;

import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.AbstractClassifier;
import classifiers.ClassifierFactory;
import classifiers.ELinearClassifier;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import evaluation.CrossValidationOutput;
import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.FeatureSelectionExperimentReport;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelector;


public class FeatureSelectionCommand implements IExperimentCommand{

	private String GraphsPath;
	private  List<EFeatureSelectionAlgorithm> selectionAlgs = null;
	private List<ELinearClassifier> classifiers = null;
	private Logger log = Util.getFileLogger(FeatureSelectionCommand.class.getName(), "./results/logs/log.txt");
	private int KmaxFeatures;
	private StopWatch timer;
	private int folds;
	private boolean tuneJustInAccuracy;

	public FeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath){
		this(classifiers,FetSelectionAlgs,graphPath,Integer.MAX_VALUE, 10,true);
	}

	public FeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath, int kMaxFeatures, int folds, boolean tuneJustInAccuracy ){

		this.GraphsPath = graphPath;
		this.selectionAlgs = FetSelectionAlgs;
		this.classifiers = classifiers;
		this.KmaxFeatures = kMaxFeatures;
		timer = new StopWatch();timer.start();
		this.folds = folds;
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

			//instatiate reports
			int maxNumFeatures = KmaxFeatures < cp.getNumAttributes() - 1 ? KmaxFeatures : cp.getNumAttributes() - 1;
			FeatureSelectionExperimentReport exp_acc = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.ACCURACY.name());
			FeatureSelectionExperimentReport exp_pre = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.PRECISION.name());
			FeatureSelectionExperimentReport exp_rec = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures,  folds, EClassificationMetric.RECALL.name());
			FeatureSelectionExperimentReport exp_fm = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures,  folds, EClassificationMetric.FSCORE.name());

			//run all selected feature selection algorithm
			for (EFeatureSelectionAlgorithm alg : this.selectionAlgs) {

				System.out.println(">>> feature selection alg: " + alg.name());
				long fetSeleTimer = timer.getTime();

				try {

					for (int n_features = 1; n_features <= maxNumFeatures; n_features++) {


						//get the filter
						FeatureSelector filter = FeatureSelector.create(alg, new FeatureSelectionFactoryParameters(n_features, cl, cp.getData()));

						//cross validate
						CrossValidationOutput outlr;
						if(this.tuneJustInAccuracy){
							outlr = eval.crossValidateModelTuneInAccuraccy(cl, filter,cp, folds, System.currentTimeMillis(), param);
						}else{
							outlr = eval.CrossValidation(cl, filter,cp, folds, System.currentTimeMillis(), param,Lists.newArrayList(EClassificationMetric.values()));

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



					}
					System.out.println(">>> " + alg.name() + " done! " + "! Elapsed time(HH:mm:ss.S): " + DurationFormatUtils.formatDuration((timer.getTime() - fetSeleTimer), "HH:mm:ss.S") );

				} catch (Exception e) {
					String msg = Joiner.on(" ").skipNulls().join("problem in:",cp.getName(),ecl.name(),alg.name());
					log.log(Level.WARNING,msg , e);
					System.out.println(msg);
				}

			}

			this.savePartialResuls(result, exp_acc, exp_pre, exp_rec, exp_fm);

			System.out.println(">> " + ecl.name() + " done! Elapsed time(HH:mm:ss.S): " 
					+ DurationFormatUtils.formatDuration(timer.getTime() - classifierTime, "HH:mm:ss.S") );
		}

		System.out.println(">Problem " + cp.getName() + " done! Acutal Simulation time(HH:mm:ss.S): " + timer.toString() );

		return result;
	}


	private void savePartialResuls( List<AbstractExperimentReport> results, AbstractExperimentReport... exps){
		for (AbstractExperimentReport exp : exps) {
			exp.plot(GraphsPath);
			results.add(exp);
		}
	}
}
