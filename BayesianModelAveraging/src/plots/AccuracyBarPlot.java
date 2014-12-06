package plots;

import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import classifiers.BMAOfNBWithGaussian;
import com.google.common.collect.Sets;
import classifiers.ClassifierFactory;
import evaluation.WekaEvaluationWrapper;
import classifiers.ELinearClassifier;

public class AccuracyBarPlot {
	public String[] m_ClassificationProblem;
	public ELinearClassifier[] m_Classifiers;
	public double[][] m_CVTable; 
	
	public AccuracyBarPlot(String[] CP,ELinearClassifier[] model){
		this.m_ClassificationProblem=CP.clone();
		this.m_Classifiers=model.clone();
		m_CVTable=new double[m_ClassificationProblem.length][m_Classifiers.length];
	}

	public void getTable()
	{
		try{
		for(int i=0;i<m_ClassificationProblem.length;i++){
			for(int j=0; j<m_Classifiers.length;j++){
				ClassifierFactory factory=new ClassifierFactory();
				AbstractClassifier classifier=factory.createClassifier(m_Classifiers[j]);
				ClassificationProblem cp = new ClassificationProblem(m_ClassificationProblem[i]);
				WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
				Map<String,Set<String>> paramLR = factory.getDefaultClassifiersParameters(m_Classifiers[j]);
				eval.CrossValidation(classifier, null,cp, 10, 10, paramLR,null);
				m_CVTable[i][j]=1-eval.accuracy();
			}
		}
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	public void plotTable()
	{
		DecimalFormat df = new DecimalFormat("######0.000");
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
				line+="&"+df.format(m_CVTable[i][j]*100)+"+/-"+df.format(m_CVTable[i][j]*(1-m_CVTable[i][j])*100);
			}
			line+="\\\\\n";
			OutputHead+=line;
		}
		OutputHead+="\\end{tabular}\\end{table}";
		System.out.print(OutputHead);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
