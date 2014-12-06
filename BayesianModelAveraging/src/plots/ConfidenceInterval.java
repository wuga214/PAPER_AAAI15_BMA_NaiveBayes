package plots;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import classifiers.BMAOfNBWithGaussian;

import com.google.common.collect.Sets;

import classifiers.ClassifierFactory;
import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import classifiers.ELinearClassifier;
/**
 * 
 * @author wuga
 * confidence interval table generator
 * this class is used to generate *.tex table source file
 * the output may not suitable for your latex version
 * slightly changes of this may generate what you want
 */
public class ConfidenceInterval {
	public String[] m_ClassificationProblem;
	public ELinearClassifier[] m_Classifiers;
	public double[][] m_CVTable; 
	public double[] m_DataSize;
	
	public ConfidenceInterval(String[] CP,ELinearClassifier[] model){
		this.m_ClassificationProblem=CP.clone();
		this.m_Classifiers=model.clone();
		m_CVTable=new double[m_ClassificationProblem.length][m_Classifiers.length];
		m_DataSize=new double[m_ClassificationProblem.length];
	}

	public void getTable()
	{
		try{
		for(int i=0;i<m_ClassificationProblem.length;i++){
			System.out.println("Processing classification problems...");
			for(int j=0; j<m_Classifiers.length;j++){
				ClassifierFactory factory=new ClassifierFactory();
				AbstractClassifier classifier=factory.createClassifier(m_Classifiers[j]);
				ClassificationProblem cp = new ClassificationProblem(m_ClassificationProblem[i]);
				WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
				Map<String,Set<String>> paramLR = factory.getDefaultClassifiersParameters(m_Classifiers[j]);
				List<EClassificationMetric> metric=new ArrayList<EClassificationMetric>();
				metric.add(EClassificationMetric.ACCURACY);
				eval.CrossValidation(classifier, null,cp, 10, 10, paramLR,metric);
				m_CVTable[i][j]=1-eval.accuracy();
				if(j==0){m_DataSize[i]=cp.getData().numInstances();}
			}
		}
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void plotLinear()
	{
		DecimalFormat df = new DecimalFormat("######0.00");
		for(int j=0;j<m_CVTable[0].length;j++)
		{
			String output="";
			for(int i=0;i<m_CVTable.length;i++)
			{
				String line=m_ClassificationProblem[i].substring(9, m_ClassificationProblem[i].indexOf(".arff"));
				output+="("+i+","+df.format(-(m_CVTable[i][j]-m_CVTable[i][1])*100)+")";
			}
			System.out.println(output);
		//	OutputHead+=line;
		}
	}
	
	public void plotTable()
	{
		DecimalFormat df = new DecimalFormat("######0.00");
		String OutputHead="\\begin{table}[H]\n\\caption{$N=10$ $95%$ Confidence Interval}\n\\centering\n";
		String Column="";
		String ClassifierNames="Problems";
		for(int i=0;i<m_Classifiers.length;i++){
			Column+="c ";
			ClassifierNames+="&"+m_Classifiers[i].toString();
		}
		OutputHead+="\\begin{tabular}{"+Column+"}\n\\\\\n\\hline\n";
		OutputHead+=ClassifierNames+"\\\\[0.5ex]\n";
		for(int i=0;i<m_CVTable.length;i++)
		{
			String line=m_ClassificationProblem[i].substring(9, m_ClassificationProblem[i].indexOf(".arff"));
			for(int j=0;j<m_CVTable[0].length;j++)
			{
				line+="&"+df.format(m_CVTable[i][j]*100)+"\\pm"+df.format(2.228*Math.sqrt(m_CVTable[i][j]*(1-m_CVTable[i][j])/m_DataSize[i])*100);
			}
			line+="\\\\\n";
			OutputHead+=line;
		}
		OutputHead+="\\end{tabular}\\end{table}";
		System.out.print(OutputHead);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String[] FILES;
		FILES=new String[19];
		FILES[0]="data/UCI/anneal.arff";
		FILES[1]="data/UCI/autos.arff";
		FILES[2]="data/UCI/vote.arff";
		FILES[3]="data/UCI/sick.arff";
		FILES[4]="data/UCI/crx.arff";
		FILES[5]="data/UCI/mushroom.arff";
		FILES[6]="data/UCI/heart-statlog.arff";
		FILES[7]="data/UCI/segment.arff";
		FILES[8]="data/UCI/labor.arff";
		FILES[9]="data/UCI/vowel.arff";
		FILES[10]="data/UCI/audiology.arff";
		FILES[11]="data/UCI/iris.arff";
		FILES[12]="data/UCI/zoo.arff";
		FILES[13]="data/UCI/lymph.arff";
		FILES[14]="data/UCI/soybean.arff";
		FILES[15]="data/UCI/balance-scale.arff";
		FILES[16]="data/UCI/glass.arff";
	//	FILES[17]="data/UCI/letter.arff";
		FILES[17]="data/UCI/hepatitis.arff";	
		FILES[18]="data/UCI/haberman.arff";
		
		ELinearClassifier[] classifiers;
		classifiers=new ELinearClassifier[4];
		classifiers[0]=ELinearClassifier.C_BMA;
		//classifiers[1]=ELinearClassifier.C_NAIVE_BAYES;
		classifiers[1]=ELinearClassifier.NAIVE_BAYES;
		classifiers[2]=ELinearClassifier.LOGISTIC_REGRESSION;
		classifiers[3]=ELinearClassifier.SVM_LINEAR;
		ConfidenceInterval cf=new ConfidenceInterval(FILES,classifiers);
		cf.getTable();
		cf.plotLinear();
	//	cf.plotTable();
	}

}
