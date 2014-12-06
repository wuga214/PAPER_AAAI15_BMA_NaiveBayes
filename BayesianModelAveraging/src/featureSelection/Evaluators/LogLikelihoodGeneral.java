package featureSelection.Evaluators;

import java.util.Arrays;
import java.util.BitSet;

import weka.core.Instance;

public class LogLikelihoodGeneral extends OurBaseFeatureSelectionEvaluator
{

	private static final long serialVersionUID = 7958789732385678394L;
	private int k, n;
	private double[][] cache;
	private BitSet bs;
	private Instance datum;
	
	@Override
	public double evaluateSubset( BitSet bs )
	{
		//System.out.println( Utils.bitSetToString( bs, dataDiscretized.numAttributes()-1 ) );
		k=bs.cardinality(); // the number of selected features. this makes n a variable as well
		n=k/2; // SCOTT: for some experiments, we'll need n to range from 1..k, OK to hardcode for initial tests
		this.bs=bs;
		double score=0;
		cache=new double[k+1][n+1]; // SCOTT: could move outside for loop, n and k never change within loop
		for( int i=0 ; i<dataDiscretized.size() ; i++ )
		{
			Instance datum=dataDiscretized.get( i );
			this.datum=datum;
			for( int j=0 ; j< k+1 ; j++ ) // SCOTT: actually n+1 not n according to initialization above, but not sure if it matters
				Arrays.fill( cache[j], -1 );

			// log version would use: score+= Math.log( query( 1, n, bs.nextSetBit( 0 ) ) );
			score+=Math.log(query( 1, n, bs.nextSetBit( 0 ) ));
		}
		return score;
	}
	
	private double query( int i, int j, int f ) // SCOTT: i is index of feature current examining, j is remaining features needed to be true, f is actual feature index for i^th feature in use
	{
		// SCOTT: base cases
		if( j==0 ) // SCOTT: at least n features have been set to true... prob of count satisfaction from here out is 1 since leads to unrestricted marginal over LHS of conditional prob
			return 1;
		if( f==-1 || j>k-i+1 ) // SCOTT: j is not 0 and we've exhausted available features so prob of count satisfaction from here out is 0, don't think +1 is needed for second case -- fewer than j features left so cannot satisfy count constraint, but cannot hurt (just 1 more step of recursion)
			return 0;
		if( cache[i][j]!=-1 ) // SCOTT: cached!
			return cache[i][j];

		// SCOTT: recursive computation... note that 1 is hardcoded true and 0 hardcoded false... does Rodrigo enforce this, or have a different way to access "true" and "false" indices?
		double trueCase=probs[f][trueLabel][(int)datum.value(f)]*query( i+1, j-1, bs.nextSetBit( f+1 ) );
		double falseCase=probs[f][negLabel][(int)datum.value(f)]*query( i+1, j, bs.nextSetBit( f+1 ) );

		// SCOTT: Need to memoize cached result, otherwise exponential time!
		double sum = trueCase + falseCase;

		cache[i][j] = sum;
		return sum;
	}
}