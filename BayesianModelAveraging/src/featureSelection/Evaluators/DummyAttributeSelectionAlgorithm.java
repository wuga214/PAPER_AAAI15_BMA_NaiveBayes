package featureSelection.Evaluators;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

//class to implement a attribute selection algorithm that analysis each feature idependently
public class DummyAttributeSelectionAlgorithm extends ASEvaluation implements AttributeEvaluator {

	private Instances data;
	
	public DummyAttributeSelectionAlgorithm(){
		super();
	}

	//used to initialize the Attribute selection algorithm
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		this.data = data;
		
	}

	//calculate the rank of a single attribute
	@Override
	public double evaluateAttribute(int idx) throws Exception {
		return idx;
	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}

	
}
