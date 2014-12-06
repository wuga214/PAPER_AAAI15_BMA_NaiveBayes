package featureSelection.Evaluators;

import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;

import utils.IConstants;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingWithUserConstant;

import com.google.common.base.Joiner;

public abstract class OurBaseFeatureSelectionEvaluator extends ASEvaluation implements SubsetEvaluator  {


	private static final long serialVersionUID = 4689739357973531768L;

	protected Instances dataDiscretized;

	//probs[i][yd][xi] = P(y_i| xi, fi)
	protected double[][][] probs;

	//count event y=yd, xi = xd for fi
	protected double[][][] count_fiyx;

	//count event xi = xd for fi
	protected double[][] count_fix;

	//laplace smoth paramter
	protected double lspValue = 1;

	//true label of the data set
	protected Integer trueLabel = null;

	//negative label
	protected Integer negLabel = null;


	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {

		if( data.numClasses()>2 )
			throw new IllegalArgumentException( "Binary classification only." );

		if(!IConstants.getInstance().getDataSet2TrueLabels().containsKey(data.relationName()))
			throw new IllegalArgumentException( "True label of the data set was not indicated" );
		else
			this.trueLabel = IConstants.getInstance().getDataSet2TrueLabels().get(data.relationName());

		for(int yindex = 0 ; yindex < data.numClasses(); yindex++){
			if(yindex != this.trueLabel){
				negLabel = yindex;
				break;
			}
		}

		//discretize the data set
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);

		//dealling with missing values
		Set<Integer> attibutesToAddMissing = new HashSet<Integer>();
		for (int attIndex = 0; attIndex < this.dataDiscretized.numAttributes() - 1 ; attIndex++) {
			double[] attributeValues = this.dataDiscretized.attributeToDoubleArray(attIndex);
			for (int d = 0; d < this.dataDiscretized.numInstances(); d++) {
				if(Double.isNaN(attributeValues[d])){
					attibutesToAddMissing.add(attIndex + 1);
					break;
				}
			}
		}

		ReplaceMissingWithUserConstant filter = new ReplaceMissingWithUserConstant();
		String t =  Joiner.on(", ").skipNulls().join(attibutesToAddMissing);
		filter.setAttributes(t);
		filter.setNominalStringReplacementValue("'?'");
		filter.setInputFormat(this.dataDiscretized);
		this.dataDiscretized = Filter.useFilter(dataDiscretized, filter);


		//variables to help in the vector instatiations
		int numberOfAttributes = this.dataDiscretized.numAttributes() - 1;
		int numberOfClasses = this.dataDiscretized.numClasses();
		int multplier = numberOfClasses;

		//instatiation
		this.count_fiyx = new double[numberOfAttributes][numberOfClasses][];
		this.probs = new double [numberOfAttributes][numberOfClasses][];
		this.count_fix = new double[numberOfAttributes][];

		for (int fi = 0; fi < numberOfAttributes; fi++) {	
			int xvalue = this.dataDiscretized.attribute(fi).numValues();
			this.count_fix[fi] = new double[xvalue]; 

			for(int yvalue = 0 ; yvalue < numberOfClasses; yvalue++){
				this.count_fiyx[fi][yvalue] = new double[xvalue];
				this.probs[fi][yvalue] = new double[xvalue]; 
			}
		}

		//counts
		for(int fi = 0; fi < this.count_fiyx.length; fi++){

			//Gets the value of all instances for the attribute f  
			double[]  fAttibuteValues = this.dataDiscretized.attributeToDoubleArray(fi);
			double[] ydLabels = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());

			for (int d = 0; d < fAttibuteValues.length; d++ ) {
				int xdi = (int)fAttibuteValues[d];
				int yd = (int)ydLabels[d];

				this.count_fix[fi][xdi]++;
				this.count_fiyx[fi][yd][xdi]++;

			}
		}
		//probability computation
		for (int fi = 0; fi < this.probs.length; fi++) {
			for (int y = 0; y < this.probs[fi].length; y++) {
				for (int x = 0; x < this.probs[fi][y].length; x++) {
					this.probs[fi][y][x] = (this.count_fiyx[fi][y][x] + lspValue)/(this.count_fix[fi][x] + lspValue * multplier);
				}
			}
		}

		/*//debug - this properties have to be true unless you create another math
		for (int fi = 0; fi < this.probs.length; fi++) {

			for(int y1 = 0; y1 < this.probs[y1].length; y1++){
				for (int x = 0; x < this.probs[fi][y1].length; x++) {
					double debug = 0;

					for (int y = 0; y < this.probs[fi].length; y++) {
						debug+= this.probs[fi][y][x];
					}

					System.out.println("Have to be 1: " + debug);
				}
			}
		}*/	
	}
}