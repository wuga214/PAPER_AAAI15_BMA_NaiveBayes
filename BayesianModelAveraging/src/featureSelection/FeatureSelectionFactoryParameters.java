package featureSelection;

import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.primitives.Ints;

import weka.attributeSelection.ASEvaluation;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class FeatureSelectionFactoryParameters {

	//number of features
	private int numberOfFeature;
	
	//base classifier for wrrapers
	private AbstractClassifier classifier;
	
	//data where the feature algorithm will work to get header information
	private Instances formatData;
	
	//Start set of attributes to optimize the search
	private int[] featureStartSet;
	
	//this is to speed up the code
	private ASEvaluation prebuiltEvaluator;

	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData) {
		this(numberOfFeature, classifier,formatData,null,null);
	}
	
	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData, int[] featureStartSet ) {
		this(numberOfFeature, classifier,formatData, featureStartSet ,null);
	}
	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData, int[] featureStartSet, ASEvaluation preBuiltEvaluator) {
		super();
		this.numberOfFeature = numberOfFeature;
		this.classifier = classifier;
		this.formatData = formatData;
		this.featureStartSet = featureStartSet;
		this.prebuiltEvaluator = preBuiltEvaluator;
	}

	//getter and setter 
	public Instances getFormatData() {
		return formatData;
	}

	public void setFormatData(Instances formatData) {
		this.formatData = formatData;
	}

	public int getNumberOfFeature() {
		return numberOfFeature;
	}

	public void setNumberOfFeature(int numberOfFeature) {
		this.numberOfFeature = numberOfFeature;
	}

	public AbstractClassifier getClassifier() {
		return classifier;
	}

	public void setClassifier(AbstractClassifier classifier) {
		this.classifier = classifier;
	}
	public int[] getFeatureStartSet() {
		return featureStartSet;
	}
	public void setFeatureStartSet(int[] featureStartSet) {
		this.featureStartSet = featureStartSet;
	}
	//this code is a workaround because ranges in weka are indexed by one not by zero
	public String featuresSelected2WekaRangeRepresentation(){
		String stringRep = "";
		
		if(this.featureStartSet != null && this.featureStartSet.length > 0){
			Integer[] valuesModified = new Integer[this.featureStartSet.length];
			for (int i = 0; i < featureStartSet.length; i++) valuesModified[i] = featureStartSet[i] + 1;
			stringRep = Joiner.on(", ").skipNulls().join(valuesModified);
		}
		
		return stringRep;
	}
	//this is a property to store evaluators built in order to avoid calculate each time that it is necessary
	public ASEvaluation getPrebuiltEvaluator() {
		return prebuiltEvaluator;
	}
	public void setPrebuiltEvaluator(ASEvaluation prebuiltEvaluator) {
		this.prebuiltEvaluator = prebuiltEvaluator;
	}
	
	
}
