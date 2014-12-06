package featureSelection.Evaluators;

import java.util.BitSet;

import JavaMI.MutualInformation;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class MRMRFeatureSelection extends ASEvaluation implements SubsetEvaluator   {
	
	private static final long serialVersionUID = -2202158496041221332L;
	private double[] relevance;
	private double[][] redundancy;
	private Instances dataDiscretized; // don't work with the data directly use a copy
	
	@Override
	public void buildEvaluator( Instances data ) throws Exception
	{
		// discretize data
		Discretize disc=new Discretize();
		disc.setUseBetterEncoding( true );
		disc.setInputFormat( data );
		dataDiscretized = Filter.useFilter( data, disc );
		
		// calculate relevance & redundancy
		int numFeatures = dataDiscretized.numAttributes()-1; // output class is an attribute in weka
		relevance=new double[numFeatures];
		redundancy=new double[numFeatures][numFeatures];
		double[] out=dataDiscretized.attributeToDoubleArray( dataDiscretized.classIndex() );
		for( int i=0 ; i<numFeatures ; i++ )
		{
			relevance[i]=MutualInformation.calculateMutualInformation( out, dataDiscretized.attributeToDoubleArray( i ) );
			double[] arr=dataDiscretized.attributeToDoubleArray( i );
			for( int j=0 ; j<redundancy[i].length ; j++ )
				redundancy[i][j]=MutualInformation.calculateMutualInformation( arr, dataDiscretized.attributeToDoubleArray( j ) );
		}
	}
	
	@Override
	public double evaluateSubset( BitSet bs )
	{
		double rel=0.0, red=0.0; // @rel=relavence, @red=redundancy
		for( int i=bs.nextSetBit( 0 ) ; i>=0 ; i=bs.nextSetBit( i+1 ) )
		{
			rel+=relevance[i];
			for( int j=bs.nextSetBit( i ) ; j>=0 ; j=bs.nextSetBit( j+1 ) )
				red+=redundancy[i][j];
		}
		return (rel/bs.cardinality())-(red/( bs.cardinality()*bs.cardinality() ));
	}
}
