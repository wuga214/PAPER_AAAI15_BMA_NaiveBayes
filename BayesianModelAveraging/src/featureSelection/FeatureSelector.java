package featureSelection;

import org.apache.commons.lang3.ArrayUtils;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import featureSelection.*;
import featureSelection.Evaluators.ConditionalEntropyFeatureSelection;
import featureSelection.Evaluators.ExpectationGeneral;
import featureSelection.Evaluators.HighPreLogLikelihoodEvaluator;
import featureSelection.Evaluators.HighPrecExpectationEvaluator;
import featureSelection.Evaluators.HighRecExpectationEvaluator;
import featureSelection.Evaluators.HighRecLogLikelihoodEvaluator;
import featureSelection.Evaluators.LogLikelihoodGeneral;
import featureSelection.Evaluators.MRMRFeatureSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SVMAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.core.Instances;
import weka.core.SelectedTag;

public class FeatureSelector {

	private ASEvaluation evaluator;
	private ASSearch search;
	private boolean isPrebuilt = false;



	public int[] selectAttributes(Instances data) throws Exception{
		int[] selectedAtributes;

		if(!isPrebuilt)
			evaluator.buildEvaluator(data);

		selectedAtributes = search.search(evaluator, data);
		
		if(search instanceof Ranker){
			Ranker r = (Ranker)search;
			int numToSelect = r.getCalculatedNumToSelect();
			selectedAtributes = ArrayUtils.subarray(selectedAtributes, 0, numToSelect);
	
		}
			
		selectedAtributes = ArrayUtils.add(selectedAtributes, data.classIndex());
		
		return selectedAtributes;
	}

	public boolean isPrebuilt() {
		return isPrebuilt;
	}

	public void setPrebuilt(boolean isPrebuilt) {
		this.isPrebuilt = isPrebuilt;
	}
	
	public ASEvaluation getEvaluator() {
		return evaluator;
	}

	public void setEvaluator(ASEvaluation evaluator) {
		this.evaluator = evaluator;

	}

	public ASSearch getSearch() {
		return search;
	}

	public void setSearch(ASSearch search) {
		this.search = search;
	}

	public static FeatureSelector create(EFeatureSelectionAlgorithm EType, 
			FeatureSelectionFactoryParameters parameter) throws Exception{
		//base classes
		FeatureSelector attributeSelection = new FeatureSelector();
		ASEvaluation evaluator = null;
		ASSearch search = null;

		//set the evaluator and searcher for each algorithm
		switch (EType) {
		//rank attributes
		case INFORMATIONGAIN_RANK:
			//mutual information evaluator
			evaluator = new InfoGainAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case CONDITIONAL_ENTROPY_RANK:
			//conditional entropy evaluator
			evaluator = new ConditionalEntropyFeatureSelection();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case CORRELATION_BASED_RANK:
			//Pearson Correlation by attribute
			evaluator = new CorrelationAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case GAINRATIO_RANK:
			//conditional entropy evaluator
			evaluator = new GainRatioAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case SYMMETRICAL_UNCERT_RANK:
			//conditional entropy evaluator
			evaluator = new SymmetricalUncertAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case RELIFF:
			//reliff evaluator
			evaluator = new ReliefFAttributeEval();
			//rank by evaluator values
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

			//subset search
		case CORRELATION_BASED_SUBSET:
			//correlation based evaluator
			evaluator = new CfsSubsetEval();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case MRMR_MI_BASED_SUBSET:
			//correlation based evaluator
			evaluator = new MRMRFeatureSelection();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case FCBF:
			//correlation based evaluaton
			evaluator = new SymmetricalUncertAttributeSetEval();
			//choose based on a graddy search 
			search = new FCBFSearch();
			((FCBFSearch)search).setNumToSelect(parameter.getNumberOfFeature());
			break;


			//wrapper approaches
		case SVMRFE:
			//svmrfe evaluator
			evaluator = new SVMAttributeEval();
			//rank by evaluator values
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;

		case FORWARD_SELECTION_WRAPPER:
			//wrraper method with the classifier
			WrapperSubsetEval wrraperEvaluator = new WrapperSubsetEval();
			wrraperEvaluator.setClassifier(parameter.getClassifier());
			wrraperEvaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_AUC,WrapperSubsetEval.TAGS_EVALUATION));
			wrraperEvaluator.setFolds(10);
			evaluator = wrraperEvaluator;

			//greedy search algorithm
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case BACKWARD_SELECTION_WRAPPER:
			//wrraper method with the classifier
			WrapperSubsetEval BCKwrraperEvaluator = new WrapperSubsetEval();
			BCKwrraperEvaluator.setClassifier(parameter.getClassifier());
			BCKwrraperEvaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_AUC,WrapperSubsetEval.TAGS_EVALUATION));
			BCKwrraperEvaluator.setFolds(10);
			evaluator = BCKwrraperEvaluator;

			//greedy search algorithm
			search = new MyGreedySearch();
			((MyGreedySearch)search).setSearchBackwards(true);			
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case HIGH_PRE_EXPECT_APP:
			//high precision evaluator derived from the maximization of the expectation 
			evaluator = new HighPrecExpectationEvaluator();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case HIGH_PRE_LOGLIK_APP:
			//high precision evaluator derived from the maximization of the expectation 
			evaluator = new HighPreLogLikelihoodEvaluator();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case HIGH_REC_EXPECT_APP:
			//high precision evaluator derived from the maximization of the expectation 
			evaluator = new HighRecExpectationEvaluator();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;

		case HIGH_REC_LOG_APP:
			//high precision evaluator derived from the maximization of the expectation 
			evaluator = new HighRecLogLikelihoodEvaluator();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;
			
		case EXPECTATION_GENERAL:
			//n out of k approach using expectation
			evaluator = new ExpectationGeneral();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;
			
		case LOGLIKELIHOOD_GENERAL:
			//n out of k approach using loglikelihood
			evaluator = new LogLikelihoodGeneral();
			//choose based on a graddy search 
			search = new MyGreedySearch();
			((MyGreedySearch)search).setNumToSelect(parameter.getNumberOfFeature());
			if(parameter.getFeatureStartSet() != null && parameter.getFeatureStartSet().length > 0 ){
				((MyGreedySearch)search).setStartSet(parameter.featuresSelected2WekaRangeRepresentation());
			}
			break;
		}

		//code to reuse the evaluators
		if(parameter.getPrebuiltEvaluator() != null){
			evaluator = parameter.getPrebuiltEvaluator();
			attributeSelection.setPrebuilt(true);
		}

		//wrap the evaluator and the search algorithm
		attributeSelection.setEvaluator(evaluator);
		attributeSelection.setSearch(search);
		//attributeSelection.setInputFormat(parameter.getFormatData());

		//return the filter
		return attributeSelection;
	}

	







}
