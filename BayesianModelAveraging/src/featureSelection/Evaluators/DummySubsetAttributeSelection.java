package featureSelection.Evaluators;

import java.util.BitSet;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;

public class DummySubsetAttributeSelection extends ASEvaluation implements SubsetEvaluator  {

	private Instances data;
	
	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		this.data = data;

	}

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {
		
		for (int i = 0; i < this.data.numAttributes(); i++) {
			if(subSet.get(i)){
				return Math.random();
			}
		}
		return 0;
	}
	
	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}

}
