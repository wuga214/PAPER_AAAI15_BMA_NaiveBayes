package utils;
import weka.core.Attribute;
import weka.core.Statistics;
/**
 * 
 * @author wuga
 * This is density estimator!
 * storage unit after training!
 * nominal and numeric features can be stored through different mechanism
 * Gnode is declared in other file! 
 */
public class ProbabilityGenerator {
	public Gnode[] valuecounter; //build counter and probability storage for each possible variable of an attribute 
	public int totalcounter;    //total number of variables under certain attribute && class
	public double expectation; //E[X]
	public double squaredexpectation; //E[x^2]
	public int discretized=1;
	public double mean;
	public double variance;
	public int classnumber;
	private double m_Precision;
	public double m_StandardDev;
	
	public ProbabilityGenerator(Attribute attr, int classes)
	{
		if(attr.type()!=0)
		{
			valuecounter=new Gnode[attr.numValues()];
			for(int i=0;i<attr.numValues();i++)
			{
				valuecounter[i]=new Gnode();
				valuecounter[i].name=attr.value(i).toString();
			}
			totalcounter=attr.numValues();
		}
		else
		{
		expectation=0;
		squaredexpectation=0;
		discretized=0;
		totalcounter=0;
		classnumber=classes;
		}
	}
	
	public ProbabilityGenerator(Attribute attr, int classes, double precision)
	{
		if(attr.type()!=0)
		{
			valuecounter=new Gnode[attr.numValues()];
			for(int i=0;i<attr.numValues();i++)
			{
				valuecounter[i]=new Gnode();
				valuecounter[i].name=attr.value(i).toString();
			}
			totalcounter=attr.numValues();
		}
		else
		{
		expectation=0;
		squaredexpectation=0;
		discretized=0;
		totalcounter=0;
		classnumber=classes;
		m_Precision=precision;
		m_StandardDev=m_Precision/(2*3);
		mean=0;
		}
	}
	//counting..
	public void addValue(String name)
	{
		
		for(Gnode temp:valuecounter)
			{if(temp.name==name)
				{
				temp.counter++;
				}
			}
		totalcounter++;		
	}
	
	public void addValue(double value)
	{
		value=round(value);
		expectation+=value;
		squaredexpectation+=value*value;
		//squaredexpectation+=Math.pow(value,2);
		totalcounter++;

	}
	
	private double round(double value)
	{
		return Math.rint(value/m_Precision)*m_Precision;
	}
	
	public void eraseValue(String name)
	{
		for(Gnode temp:valuecounter)
		{if(temp.name==name)
			{
			temp.counter--;
			}
		}
	totalcounter--;
		
	}
	
	public void cleanup()
	{
		for(Gnode temp:valuecounter)
		{
			temp.counter=0;
		}
	totalcounter=0;		
	}
	
	//log form
	public double checkClassPobability(int classNumber)
	{
		return valuecounter[classNumber].probability;
	}
	//original form
	public double OcheckClassPobability(int classNumber)
	{
		return valuecounter[classNumber].Oprobability;
	}
	//original form
	public double OcheckPobability(double x)
	{
		x=round(x);
		double temp1=Statistics.normalProbability((x-mean-(m_Precision/2))/m_StandardDev);
		double temp2=Statistics.normalProbability((x-mean+(m_Precision/2))/m_StandardDev);
	//	double hight=densityCalculator((x-mean)/m_StandardDev);
		return Math.max(1e-75,temp2-temp1);
	//	return Math.max(1e-75, hight*m_Precision);
	}
	
	public double densityCalculator(double x)
	{
		return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI);
	}
	//log form
	public double checkPobability(double x)
	{
		return Math.log(OcheckPobability(x));
	}
	//the last step of training process, calculate LogP(C) OR LogP(X|C) and store the result, ready for usage.
	public void probabilityCal()
	{
		if(discretized!=0)
		{
			for(Gnode temp:valuecounter)
			{
				temp.probability=Math.log(((double)temp.counter)/totalcounter);
			}
		}else
		{
			if(totalcounter>0)
			{
			mean=expectation/totalcounter;
			double temp=squaredexpectation/totalcounter;
		//	variance=Math.abs(temp-Math.pow(mean,2));
			variance=Math.abs(temp-mean*mean);
			double stdDev=Math.sqrt(variance);
			if(stdDev>1e-10)
			{
				m_StandardDev=Math.max(m_Precision/(2*3), stdDev);
			}
			}
		}
	}
	
	//original form for nominal features
	public void probability()
	{
		if(discretized!=0)
		{
			for(Gnode temp:valuecounter)
			{
				temp.Oprobability=((double)temp.counter)/totalcounter;
			}
		}else
		{
		/*	mean=expectation/totalcounter;
			double temp=squaredexpectation/totalcounter;
			variance=temp-Math.pow(mean,2);
			double stdDev=Math.sqrt(variance);
			if(stdDev>1e-10)
			{
				m_StandardDev=Math.max(m_Precision/(2*3), stdDev);
			}*/
		}
		
	}
}
