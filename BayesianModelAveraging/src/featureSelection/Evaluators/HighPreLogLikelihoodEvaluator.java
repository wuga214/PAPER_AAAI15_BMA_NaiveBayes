package featureSelection.Evaluators;

import java.util.BitSet;

import weka.core.Instance;

public class HighPreLogLikelihoodEvaluator extends OurBaseFeatureSelectionEvaluator{

	private static final long serialVersionUID = 8844373627809926374L;

	@Override
	public double evaluateSubset( BitSet bs ) throws IllegalArgumentException
	{
		double score=0;
		for( Instance datum : dataDiscretized )
		{
			double p=1;
			for( int i=bs.nextSetBit( 0 ) ; i>=0 ; i=bs.nextSetBit( i+1 ) ) // for each selected feature @i
				p*=probs[i][trueLabel][(int)datum.value(i)];
			score+=( datum.classValue() == trueLabel ) ? Math.log(p) : Math.log(1 - p);
		}
		return score;
	}
}
