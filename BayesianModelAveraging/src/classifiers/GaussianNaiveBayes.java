package classifiers;

import utils.ProbabilityGenerator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;
import weka.classifiers.Evaluation;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.Sets;

import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import problems.ClassificationProblem;
public class GaussianNaiveBayes 
	extends AbstractClassifier{
	
	//Class ProbabilityGenerator is in package utils!
	private ProbabilityGenerator[/*Attribute_number*/][/*Class_number*/] m_CondProb;//P(X|C)
	private ProbabilityGenerator m_PrioProb;//P(C)
	private Discretize m_Filter;
	private boolean m_Discretization=false; 
	private double[] m_Precision;
	protected static final double DEFAULT_NUM_PRECISION=0.01;
	
	/**
	 * Discretizing support function
	 * Discretize training dataset
	 * @param instances
	 * @return discretized instances
	 * @throws Exception
	 */
	public Instances toDiscretize(Instances instances) throws Exception
	{
		m_Filter=new Discretize();
		m_Filter.setInputFormat(instances);
		Instances data=Filter.useFilter(instances, m_Filter);
		return data;
	}
	
	/**
	 * Discretizing support function
	 * Discretize testing dataset
	 * @param instance
	 * @return discretized instance
	 * @throws Exception
	 */
	public Instance toDiscretize(Instance instance) throws Exception
	{
		m_Filter.input(instance);
		Instance DiscInst=m_Filter.output();
		return DiscInst;
	}
	
	/**
	 * To calcuate probability from gaussian distribution
	 * we need precision as \delta x and multiply it with density
	 * This function used to calculate proper precision
	 * @param instances
	 */
	public void getPrecision(Instances instances)
	{
		Instances data=new Instances(instances);
		m_Precision=new double[data.numAttributes()-1];
		for(int i=0;i<data.numAttributes()-1;i++)
		{
			m_Precision[i]=DEFAULT_NUM_PRECISION;
			if(data.attribute(i).type()==Attribute.NUMERIC)
			{
				data.sort(i);
				if((data.numInstances()>0)&&!data.instance(0).isMissing(i))
						{
							double lastVal=data.instance(0).value(i);
							double currentVal,deltaSum=0;
							int distinct=0;
							for(int j=1;j<data.numInstances();j++)
							{
								if(data.instance(j).isMissing(i))
								{
									break;
								}
								currentVal=data.instance(j).value(i);
								if(currentVal!=lastVal)
								{
									deltaSum+=currentVal-lastVal;
									lastVal=currentVal;
									distinct++;
								}
							}
							if(distinct>0)
							{
								m_Precision[i]=deltaSum/distinct;
							}
						}
			}
		}
	}
	
	/**
	 * Training function
	 * Store trained data into private variables
	 *  @param instances
	 */
	public void buildClassifier(Instances instances) throws Exception {
		
		//training process
		Instances data=new Instances(instances);
		data.deleteWithMissingClass();
		if(m_Discretization)
		{
		data=toDiscretize(data);
		}
		getPrecision(data);
		m_CondProb=new ProbabilityGenerator[data.numAttributes()-1][data.numClasses()];
		for(int i=0;i<data.numClasses();i++)
		{
			for(int j=0;j<data.numAttributes()-1;j++)
			{
				m_CondProb[j][i]=new ProbabilityGenerator(data.attribute(j),data.numClasses(),m_Precision[j]);
			}
		}
		m_PrioProb=new ProbabilityGenerator(data.classAttribute(),data.numClasses());

		for(int i=0;i<data.numInstances();i++)
		{
			Instance currentInst=data.instance(i);
			for(int j=0;j<data.numAttributes()-1;j++)
			{
				if(!currentInst.isMissing(j))
				{
				if(data.attribute(j).type()!=0)
				{
					m_CondProb[j][(int)currentInst.classValue()].addValue(currentInst.attribute(j).value((int)currentInst.value(j)));
				}
				else
				{
					m_CondProb[j][(int)currentInst.classValue()].addValue(currentInst.value(j));
				}
				}
				
			}
			m_PrioProb.addValue(currentInst.classAttribute().value((int)currentInst.classValue()));
		}
		
		for(int j=0;j<data.numAttributes()-1;j++)
		{
			for(int i=0;i<data.numClasses();i++)
			{
				m_CondProb[j][i].probabilityCal();
			}
		}
		m_PrioProb.probabilityCal();
		}
	
	/**
	 * testing process!
	 * @param instance
	 * @return predicted class label
	 */
	public double classifyInstance(Instance instance) throws Exception {
		//testing process
		double[/*Class_Number*/] prob;
		double selectedClassIndex=0;
		double maxProbability=0;
		Instance DiscInst=instance;
		if(m_Discretization)
		{
			DiscInst=toDiscretize(DiscInst);
		}
		
		prob=new double[DiscInst.numClasses()];
		for(int i=0;i<DiscInst.numClasses();i++)
		{
			prob[i]=0;
			for(int j=0;j<DiscInst.numAttributes()-1;j++)
			{
				if(!DiscInst.isMissing(j))
				{
				
				if(DiscInst.attribute(j).type()!=0)
				{
					prob[i]+=m_CondProb[j][i].valuecounter[(int)DiscInst.value(j)].probability;
				}
				else
				{
					prob[i]+=m_CondProb[j][i].checkPobability(DiscInst.value(j));
				}
				}
				
			}
			prob[i]+=m_PrioProb.checkClassPobability(i);
			if(i==0){maxProbability=prob[i];}
			if(i>0){
				if(maxProbability<prob[i])
				{
					maxProbability=prob[i];
					selectedClassIndex=i;}
			}
			
		}
		return selectedClassIndex;
		
	}
	
	/**
	 * Options
	 * D is disretization switcher
	 */
	public void setOptions(String[] options) throws Exception {
		//work around to avoid the api print trash in the console
		if(options.length!=0)
		{
		if(options[0]=="-D")
		{m_Discretization=true;}
		}
	}
	
	/**
	 * Main function
	 * the comments below are optional commands
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try {
		//	ClassificationProblem cp = new ClassificationProblem("data/autos.arff");
			AbstractClassifier classifier = new GaussianNaiveBayes();
			ClassificationProblem cp = new ClassificationProblem("./data/NEWS/news.arff");
			List<EClassificationMetric> metric=new ArrayList<EClassificationMetric>();
			metric.add(EClassificationMetric.ACCURACY);
	//		long startTime = System.nanoTime();
	//		classifier.buildClassifier(cp.getData());
	//		classifier.distributionForInstance(cp.getData().instance(0));
			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
			
			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			eval.CrossValidation(classifier, null,cp, 10, 10, null,metric);
	//		long endTime = System.nanoTime();
	//		long duration = endTime - startTime;
			System.out.println(eval.accuracy());
	//		System.out.println(duration);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
 
}
