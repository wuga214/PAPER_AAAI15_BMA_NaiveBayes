package evaluation;


import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;

import problems.ClassificationProblem;
import utils.IConstants;
import utils.Util;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import com.google.common.collect.Lists;

import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelector;

public class WekaEvaluationWrapper{

	private Evaluation wekaEvaluation ;
	private ClassificationProblem problem;

	public WekaEvaluationWrapper(ClassificationProblem cp){
		this.problem = cp;
	}

	//this method is to speed up the simulation. it can tune a model in each metric and report each metric
	public CrossValidationOutput fastCrossValidation(AbstractClassifier c, Instances randData, int numOfFeaturesToSelect, 
			EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params, int[][] previousFeature, ASEvaluation[] fold2Evaluators, List<EClassificationMetric> metrics)throws Exception{

		//output
		CrossValidationOutput cvo = new CrossValidationOutput(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {

			//fold results
			FoldResult fr = new FoldResult();


			///split: (train + validation) and test, train and valid
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			Instances valid = randData.testCV(folds, (n+1)%folds);
			Instances train = this.getTrainDataToTuneModel(randData, folds, n);

			//perform the feature selection in the train data
			featureSelection.FeatureSelector FeatureSelector = featureSelection.FeatureSelector.create(
					efsa, new FeatureSelectionFactoryParameters(numOfFeaturesToSelect, c, randData ,previousFeature[n], fold2Evaluators[n]));

			//select feature algorithm invocation
			int[] selectedFeatures = FeatureSelector.selectAttributes(train);

			//save selected folds
			fr.addSelectedFeature(ArrayUtils.remove(selectedFeatures,selectedFeatures.length -1));
			fr.setEvaluatorBuilt(FeatureSelector.getEvaluator());

			//build a filter to remove not selected features
			Remove rm = new Remove();
			rm.setInvertSelection(true);
			rm.setAttributeIndicesArray(selectedFeatures);
			rm.setInputFormat(train);

			//remove not selected features
			train = Filter.useFilter(train, rm);
			valid = Filter.useFilter(valid, rm);
			test = Filter.useFilter(test, rm);
			trainAndValid  = Filter.useFilter(trainAndValid, rm);

			if(metrics != null && metrics.size() > 0){

				for (EClassificationMetric metric : metrics) {

					//tune the model
					String optimumSetting = tuneClassifier(c, params, metric,
							train, valid);

					//train with optimum parameters
					c.setOptions(Utils.splitOptions(optimumSetting));
					c.buildClassifier(trainAndValid);

					this.evaluateModel(c, test);
					double metricReportValue = this.computeTunableMetric(metric);

					///workaround to save all metrics
					fr.setOptimalSetting( metric +": " + optimumSetting + " " + (fr.getOptimalSetting() != null ? fr.getOptimalSetting():""));
					fr.setMetricReported(metric, metricReportValue);

				}
			}else{

				//tune the model to find the optimal parameters given a metric
				String optimumSetting = tuneClassifier(c, params, EClassificationMetric.ACCURACY, train, valid);

				//train with optimum parameters
				c.setOptions(Utils.splitOptions(optimumSetting));
				c.buildClassifier(trainAndValid);

				//test and report the performance
				this.evaluateModel(c, test);

				//report in all metrics
				fr.setOptimalSetting(optimumSetting);
				fr.setMetricReported(EClassificationMetric.ACCURACY, this.computeTunableMetric(EClassificationMetric.ACCURACY));
				fr.setMetricReported(EClassificationMetric.PRECISION, this.computeTunableMetric(EClassificationMetric.PRECISION));
				fr.setMetricReported(EClassificationMetric.RECALL, this.computeTunableMetric(EClassificationMetric.RECALL));
				fr.setMetricReported(EClassificationMetric.FSCORE, this.computeTunableMetric(EClassificationMetric.FSCORE));

			}

			cvo.addFoldResult(fr);
		}

		return cvo;

	}

	//this method is fast but tune the model in just one metric and report it
	public CrossValidationOutput fastCrossValidationJustOneMetric(AbstractClassifier c, Instances randData, int numOfFeaturesToSelect,
			EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params, int[][] previousFeature, ASEvaluation[] fold2Evaluators,EClassificationMetric metric)throws Exception{
		return fastCrossValidation(c, randData, numOfFeaturesToSelect, efsa, folds, params, previousFeature, fold2Evaluators, Lists.newArrayList(metric));
	}

	//this method is fast but tune the model in accuracy and report all metrics
	public CrossValidationOutput fastCrossValidationTuneInAccuracy(AbstractClassifier c, Instances randData, 
			int numOfFeaturesToSelect, EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params,
			int[][] previousFeature, ASEvaluation[] fold2Evaluators)throws Exception{

		return fastCrossValidation(c, randData, numOfFeaturesToSelect, efsa, folds, params, previousFeature, fold2Evaluators, null);
	}

	
	//*******slow cross validation
	//this method tune the model in each metric and report each metric
	public CrossValidationOutput CrossValidation(AbstractClassifier c, FeatureSelector FeatureSelector,ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params, List<EClassificationMetric> metrics )throws Exception{


		CrossValidationOutput cvo = new CrossValidationOutput(seed, folds);


		Random rand = new Random(seed); 
		//randomize the data
		Instances randData = new Instances(cp.getData());   
		randData.randomize(rand);

		//if it's necessary stratify to put aproximadetly 
		//the same amount of data of different classes in each fold
		//randData.stratify(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {

			//measurement of the fold
			FoldResult foldResult = new FoldResult();

			///split: (train + validation) and test, train and valid
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			Instances valid = randData.testCV(folds, (n+1)%folds);
			Instances train = this.getTrainDataToTuneModel(randData, folds, n);

			//perform the feature selection in the train data
			if (FeatureSelector != null) {
				//get attributes indexes
				int[] selectedFeatures = FeatureSelector.selectAttributes(train);
				//build a filter to remove not selected features
				Remove rm = new Remove();
				rm.setInvertSelection(true);
				rm.setAttributeIndicesArray(selectedFeatures);
				rm.setInputFormat(train);
				//remove not selected features
				train = Filter.useFilter(train, rm);
				valid = Filter.useFilter(valid, rm);
				test = Filter.useFilter(test, rm);
				trainAndValid  = Filter.useFilter(trainAndValid, rm);
			}	

			if(metrics != null && metrics.size()>0){

				for (EClassificationMetric metric : metrics) {
					//tune the model to find the optimal parameters given a metric
					String optimumSetting = tuneClassifier(c, params, metric, train, valid);

					//train with optimum parameters
					c.setOptions(Utils.splitOptions(optimumSetting));
					c.buildClassifier(trainAndValid);

					//test and report the performance
					this.evaluateModel(c, test);
					double metricReportValue = this.computeTunableMetric(metric);


					foldResult.setMetricReported(metric, metricReportValue);
				}
				
			}else{
				
				//tune the model to find the optimal parameters in accuraccy
				String optimumSetting = tuneClassifier(c, params, EClassificationMetric.ACCURACY, train, valid);

				//train with optimum parameters
				c.setOptions(Utils.splitOptions(optimumSetting));
				c.buildClassifier(trainAndValid);

				//test and report the performance
				this.evaluateModel(c, test);
				foldResult.setMetricReported(EClassificationMetric.ACCURACY, this.computeTunableMetric(EClassificationMetric.ACCURACY));
				//foldResult.setMetricReported(EClassificationMetric.PRECISION, this.computeTunableMetric(EClassificationMetric.PRECISION));
				//foldResult.setMetricReported(EClassificationMetric.RECALL, this.computeTunableMetric(EClassificationMetric.RECALL));
				//foldResult.setMetricReported(EClassificationMetric.FSCORE, this.computeTunableMetric(EClassificationMetric.FSCORE));
			}
			
			
			cvo.addFoldResult(foldResult);
		}

		return cvo;
	}

	//this method tune the model for one metric and report it
	public CrossValidationOutput CrossValidationJustOneMetric(AbstractClassifier c, FeatureSelector FeatureSelector,ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params, EClassificationMetric metric )throws Exception{
		return CrossValidation(c, FeatureSelector, cp, folds, seed, params, Lists.newArrayList(metric));
	}
	
	//This method tune the model in accuracy and report other metrics
	public CrossValidationOutput crossValidateModelTuneInAccuraccy(AbstractClassifier c, FeatureSelector FeatureSelector,ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params ) throws Exception{
		return CrossValidation(c, FeatureSelector, cp, folds, seed, params, null);

	}


	private Instances getTrainDataToTuneModel(Instances randData, int folds, int fold){
		Instances train = null;
		for (int fn = 0; fn < folds; fn++) {
			if(fn != fold && fn != (fold+1)%10){
				if(train != null)
					train.addAll(randData.testCV(folds, fn));
				else
					train = randData.testCV(folds, fn);
			}
		}

		return train;

	}

	//compute metric to tune the model to achieve the highest performance in this metric
	//these metrics are tunable to find the model that maximizes it
	private double computeTunableMetric(EClassificationMetric metric) throws Exception {
		double metricValue = 0;
		switch (metric) {
		case ACCURACY:
			metricValue = this.accuracy();
			break;
		case PRECISION:
			metricValue = this.precision();
			break;
		case RECALL:
			metricValue = this.recall();
			break;
		case FSCORE:
			metricValue = this.fMeasure();
			break;
		default:
			metricValue = this.accuracy();
			break;
		}

		return metricValue;
	}

	//method to tune the model
	private String tuneClassifier(AbstractClassifier classifier,
			Map<String, Set<String>> params, EClassificationMetric metric,
			Instances trainData, Instances validationData) throws Exception {

		//tune the model parameters
		String optimumSetting = "";
		if(params != null && !params.isEmpty()){

			List<String> modelSettings = Util.generateModels(Lists.newArrayList(params.values()));
			double maxMetric = Double.MIN_VALUE;
			for (String setting : modelSettings) {
				//set parameter, train and evaluate
				classifier.setOptions(Utils.splitOptions(setting));
				classifier.buildClassifier(trainData);
				this.evaluateModel(classifier, validationData);
				double currentMetric = this.computeTunableMetric(metric);

				if(currentMetric  > maxMetric){
					maxMetric = currentMetric;
					optimumSetting = setting;
				}
			}

		}
		return optimumSetting;
	}

	//wrraped metrics
	public double accuracy() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return 1 - this.wekaEvaluation.errorRate();
	}


	public double errorRate() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.errorRate();
	}

	public double correct() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.correct();
	}

	public double incorrect() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.incorrect();
	}

	//execute before calculate metrics
	public void evaluateModel(AbstractClassifier c, Instances test) throws Exception {
		this.wekaEvaluation = new Evaluation(this.problem.getData());
		this.wekaEvaluation.evaluateModel(c, test);
	}

	//fmeasure
	public double fMeasure() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		/*double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.fMeasure(classValue);
		}

		return retValue/numOfLabels;
*/		
		if( this.wekaEvaluation.getHeader().numClasses()>2 )
			throw new IllegalArgumentException( "Binary classification only." );
		
		if(!IConstants.getInstance().getDataSet2TrueLabels().containsKey(this.wekaEvaluation.getHeader().relationName()))
			throw new IllegalArgumentException( "True label of the data set was not indicated" );
			
		return this.wekaEvaluation.fMeasure(IConstants.getInstance().getDataSet2TrueLabels().get(this.wekaEvaluation.getHeader().relationName()));
	}

	//overload for overrall precision
	public double precision()throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		/*double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.precision(classValue);
		}

		return retValue/numOfLabels;*/
		

		if( this.wekaEvaluation.getHeader().numClasses()>2 )
			throw new IllegalArgumentException( "Binary classification only." );
		
		if(!IConstants.getInstance().getDataSet2TrueLabels().containsKey(this.wekaEvaluation.getHeader().relationName()))
			throw new IllegalArgumentException( "True label of the data set was not indicated" );
			
		return this.wekaEvaluation.precision(IConstants.getInstance().getDataSet2TrueLabels().get(this.wekaEvaluation.getHeader().relationName()));
	}

	//overload for overrall recall
	public double recall() throws Exception{
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		/*double retValue = 0.0;
		int numOfLabels = 0;

		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();

		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.recall(classValue);
		}

		return retValue/numOfLabels;*/
		

		if( this.wekaEvaluation.getHeader().numClasses()>2 )
			throw new IllegalArgumentException( "Binary classification only." );
		
		if(!IConstants.getInstance().getDataSet2TrueLabels().containsKey(this.wekaEvaluation.getHeader().relationName()))
			throw new IllegalArgumentException( "True label of the data set was not indicated" );
			
		return this.wekaEvaluation.recall(IConstants.getInstance().getDataSet2TrueLabels().get(this.wekaEvaluation.getHeader().relationName()));
	}

	public ClassificationProblem getProblem() {
		return problem;
	}





}
