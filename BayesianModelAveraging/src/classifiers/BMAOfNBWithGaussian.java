package classifiers;

import utils.ProbabilityGenerator;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;
import utils.NumericalUtils;
import evaluation.EClassificationMetric;
import java.text.DecimalFormat;   
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import com.google.common.collect.Sets;
import evaluation.WekaEvaluationWrapper;
import problems.ClassificationProblem;

/**
 * 
 * @author wuga
 * This is the experimantal implementation of Bayesian Model Averaging for Naive Bayes classifier
 * Note that for testing and keeping structure for better understanding, 
 * the classifier is not optimized for reducing time consuming.
 * For make this classifier faster, only need to move some calculation from testing process into 
 * training process such as temp2*m_C which can definitely be moved to training process.
 */
public class BMAOfNBWithGaussian 
	extends AbstractClassifier{
	
	//Class ProbabilityGenerator is in package utils!
	private ProbabilityGenerator[/*Attribute_number*/][/*Class_number*/] m_CondProb;//P(X|C)
	private ProbabilityGenerator m_PrioProb;//P(C)
	private double[] m_LogCondProbSum;
	private double[] m_LogAveCondProbSum;
	private Discretize m_Filter;
	private Instances m_Data;
	private boolean m_Discretization=false; 
	private double[] m_Precision;
	private int m_NumOfClasses;
	private int m_NumOfAttributes;
	private double m_C;
	protected static final double DEFAULT_NUM_PRECISION=0.01;	
	public static double HYPERPARAMETER=1.0;
	public static double SUPPORTPARAMETER=1.0;
	public static String CLASSFICATION_PROBLEM_FILE="data/iris.data.arff";
	
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
		m_Data=new Instances(instances);
		m_Data.deleteWithMissingClass();
		m_C=Math.log((double)1/HYPERPARAMETER); //C=\log(\frac{1}{H})
		if(m_Discretization)
		{
			m_Data=toDiscretize(m_Data);
		}
		m_NumOfClasses=m_Data.numClasses();
		m_NumOfAttributes=m_Data.numAttributes();
		getPrecision(m_Data);		
		m_CondProb=new ProbabilityGenerator[m_NumOfAttributes-1][m_NumOfClasses+1];
		for(int i=0;i<m_NumOfClasses+1;i++)
		{
			for(int j=0;j<m_NumOfAttributes-1;j++)
			{
				m_CondProb[j][i]=new ProbabilityGenerator(m_Data.attribute(j),m_NumOfClasses,m_Precision[j]);
			}
		}
		m_PrioProb=new ProbabilityGenerator(m_Data.classAttribute(),m_NumOfClasses);

		for(int i=0;i<m_Data.numInstances();i++)
		{
			Instance currentInst=m_Data.instance(i);
			for(int j=0;j<m_NumOfAttributes-1;j++)
			{
				if(!currentInst.isMissing(j))
				{
					if(m_Data.attribute(j).type()!=0)
					{
						String temp=currentInst.attribute(j).value((int)currentInst.value(j));
						m_CondProb[j][(int)currentInst.classValue()].addValue(temp);
						m_CondProb[j][m_NumOfClasses].addValue(temp);
					}
					else
					{
						double temp=currentInst.value(j);
						m_CondProb[j][(int)currentInst.classValue()].addValue(temp);
						m_CondProb[j][m_NumOfClasses].addValue(temp);
					}
					
				}
			}
			m_PrioProb.addValue(currentInst.classAttribute().value((int)currentInst.classValue()));
		}
		
		for(int j=0;j<m_NumOfAttributes-1;j++)
		{
			for(int i=0;i<m_NumOfClasses+1;i++)
			{
				m_CondProb[j][i].probabilityCal();
				m_CondProb[j][i].probability();
			}
		}
		m_PrioProb.probabilityCal();
		m_PrioProb.probability();
		
		
		logSumPreprocessing();
		}
	
	/**
	 * \sum_{i=1}{N}\log(\Pr(x_i|y_i))
	 * \f_k=1
	 * extra training procedure for BMA
	 * @param attribute index
	 * @return
	 */
	public double LogCondProbSum(int attributeindex)
	{
		double sum=0;
		if(m_Data.attribute(attributeindex).type()!=0)
		{
			for(int i=0;i<m_Data.numInstances();i++)
			{	
			if(!m_Data.instance(i).isMissing(attributeindex))
			{
			 sum+=m_CondProb[attributeindex][(int)m_Data.instance(i).classValue()].valuecounter[(int)m_Data.instance(i).value(attributeindex)].probability;					
			}
			}
		}else
		{
			for(int i=0;i<m_Data.numInstances();i++)
			{
			if(!m_Data.instance(i).isMissing(attributeindex))
			{
			 sum+=m_CondProb[attributeindex][(int)m_Data.instance(i).classValue()].checkPobability(m_Data.instance(i).value(attributeindex));
			}
			}
		}
		
		return sum;
	}
	
	/**
	 * \sum_{i=1}{N}\log(\Pr(x_i))
	 * f_k=0
	 * extra training procedure for BMA
	 * @param attribute index
	 * @return
	 */
	public double LogAveCondProbSum(int attributeindex)
	{
		double sum=0;
		if(m_Data.attribute(attributeindex).type()!=0)
		{
			for(int i=0;i<m_Data.numInstances();i++)
			{	
			if(!m_Data.instance(i).isMissing(attributeindex))
			{
			 sum+=m_CondProb[attributeindex][m_NumOfClasses].valuecounter[(int)m_Data.instance(i).value(attributeindex)].probability;					
			}
			}
		}else
		{
			for(int i=0;i<m_Data.numInstances();i++)
			{
			if(!m_Data.instance(i).isMissing(attributeindex))
			{
			 sum+=m_CondProb[attributeindex][m_NumOfClasses].checkPobability(m_Data.instance(i).value(attributeindex));
			}
			}
		}
		
		return sum;
	}
	
	/**
	 * extra training procedure for BMA
	 * select feature index and then calculate 
	 * \sum_{i=1}{N}\log(\Pr(x_i|y_i))
	 * \sum_{i=1}{N}\log(\Pr(x_i))
	 */
	public void logSumPreprocessing()
	{
		m_LogCondProbSum=new double[m_NumOfAttributes-1];
		m_LogAveCondProbSum=new double[m_NumOfAttributes-1];
		for(int i=0;i<m_NumOfAttributes-1;i++)
		{
			m_LogCondProbSum[i]=LogCondProbSum(i);
			m_LogAveCondProbSum[i]=LogAveCondProbSum(i);			
		}
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
		prob=new double[m_NumOfClasses];
		for(int i=0;i<m_NumOfClasses;i++)
		{
			DiscInst.setClassValue(i);
			prob[i]=0;
			for(int j=0;j<m_NumOfAttributes-1;j++)
			{
				if(!DiscInst.isMissing(j))
				{
					double attributeValue=DiscInst.value(j);
					if(DiscInst.attribute(j).type()!=0)
					{
						double temp1=0,temp2=0;
						temp1=m_CondProb[j][m_NumOfClasses].valuecounter[(int)attributeValue].probability;
						temp2=m_CondProb[j][m_NumOfClasses].totalcounter-(int)DiscInst.attribute(j).numValues()+1;
						prob[i]+= NumericalUtils.LogSum(m_LogAveCondProbSum[j]+temp1,m_LogCondProbSum[j]+m_CondProb[j][i].valuecounter[(int)attributeValue].probability+temp2*m_C); 
					//	prob[i]=0;
					}
					else
					{
						double temp1=0,temp2=0;
						temp1=m_CondProb[j][m_NumOfClasses].checkPobability(attributeValue);
						temp2=m_CondProb[j][m_NumOfClasses].totalcounter+1;
						prob[i]+= NumericalUtils.LogSum(m_LogAveCondProbSum[j]+temp1,m_LogCondProbSum[j]+m_CondProb[j][i].checkPobability(attributeValue)+temp2*m_C);
					//	prob[i]=0;
					}
				}
			}
			prob[i]+=m_PrioProb.checkClassPobability(i);
			if(i==0){maxProbability=prob[i];}
			if(i>0){
				if(maxProbability<prob[i])
				{
					maxProbability=prob[i];
					selectedClassIndex=i;
				}
			}
		}
		return selectedClassIndex;
		
	}
	
	/**
	 * Options
	 * H is hyper-parameter
	 * Suggested Value range {0.5..1.5}
	 * D is disretization switcher
	 */
	public void setOptions(String[] options) throws Exception {
		
		String HP=Utils.getOption('H', options);
		boolean d=Utils.getFlag('D', options);
		if(HP.length()!=0)
		{
			HYPERPARAMETER=Double.parseDouble(HP);
		}
		if(d)
		{m_Discretization=true;}
	}
	
	/**
	 * Main function
	 * the comments below are optional commands
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DecimalFormat df = new DecimalFormat("######0.000");   
		try {			
			AbstractClassifier classifier = new BMAOfNBWithGaussian();
			ClassificationProblem cp = new ClassificationProblem("./data/UCI/vote.arff");
			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
			
			List<EClassificationMetric> metric=new ArrayList<EClassificationMetric>();
			metric.add(EClassificationMetric.ACCURACY);
	//		long startTime = System.nanoTime();
	//		classifier.buildClassifier(cp.getData());
			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-Q",Sets.newHashSet("-Q"));
			paramLR.put("-H",Sets.newHashSet("-H 1.1","-H 1.2","-H 1.3","-H 1.4","-H 1.5"));
			eval.CrossValidation(classifier, null,cp, 10, 10, paramLR,metric);
	//		classifier.distributionForInstance(cp.getData().instance(0));
	//		long endTime = System.nanoTime();
	//		long duration = endTime - startTime;
			System.out.println(eval.accuracy());
	//		System.out.println(duration);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
