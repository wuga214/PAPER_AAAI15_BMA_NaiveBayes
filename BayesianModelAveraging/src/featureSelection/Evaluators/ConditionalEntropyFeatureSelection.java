package featureSelection.Evaluators;

import JavaMI.Entropy;
import JavaMI.MutualInformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class ConditionalEntropyFeatureSelection extends ASEvaluation implements AttributeEvaluator{

	private Instances dataDiscretized;
	private double[] feature2entropy;
	
	public ConditionalEntropyFeatureSelection(){
		super();
	}

	//used to initialize the Attribute selection algorithm
	@Override
	public void buildEvaluator(Instances data) throws Exception {

		//int classIndex = data.classIndex();
		//discretizing the data
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);
		
		//precomputing entropy to be fast
		this.feature2entropy = new double[this.dataDiscretized.numAttributes()];
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		for (int i = 0; i < this.dataDiscretized.numAttributes(); i++) {
			double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
			this.feature2entropy[i] = Entropy.calculateConditionalEntropy(classATT, featureI);
		}
	}

	//calculate the mutual information for a given attribute
	@Override
	public double evaluateAttribute(int idx) throws Exception {
		
		//minimize the entropy
		return -feature2entropy[idx];

	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}
}
