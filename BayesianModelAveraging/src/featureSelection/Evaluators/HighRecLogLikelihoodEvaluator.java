package featureSelection.Evaluators;

import java.util.BitSet;

import weka.core.Instance;

public class HighRecLogLikelihoodEvaluator extends OurBaseFeatureSelectionEvaluator{


	private static final long serialVersionUID = -4289899937707173414L;

	@Override
	public double evaluateSubset( BitSet bs ) throws IllegalArgumentException
	{
		double score=0;
		for( Instance datum : dataDiscretized )
		{
			double p=1;
			for( int i=bs.nextSetBit( 0 ) ; i>=0 ; i=bs.nextSetBit( i+1 ) ) // for each selected feature @i
				p *= probs[i][negLabel][(int)datum.value(i)];

			score+=( datum.classValue() == negLabel ) ? Math.log(p) : Math.log(1-p);
		}
		return score;
	}
}
