package plots;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.AbstractClassifier;
import weka.core.Utils;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import classifiers.BMAOfNBWithGaussian;
import classifiers.ELinearClassifier;
import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
/**
 * 
 * @author wuga
 * This class generate impact of hyper parameter linear plot 
 */
public class HyperParameterLinearPlot {

	public String m_ClassificationProblem;
	public AbstractClassifier m_Classifiers;
	public static double[] results;
	public HyperParameterLinearPlot(String cpaddress,AbstractClassifier classifier,String[] settings) throws Exception{
		m_ClassificationProblem=cpaddress;
		ClassificationProblem cp = new ClassificationProblem(m_ClassificationProblem);
		
		if(settings != null){
			results=new double[settings.length];
			for (int i=0;i<settings.length;i++) {
				String setting=settings[i];
				WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
				classifier.setOptions(Utils.splitOptions(setting));
				List<EClassificationMetric> metric=new ArrayList<EClassificationMetric>();
				metric.add(EClassificationMetric.ACCURACY);
				eval.CrossValidation(classifier, null,cp, 10, 10, null,metric);
				results[i]=eval.accuracy();
			}
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DecimalFormat df = new DecimalFormat("######0.0");
		DecimalFormat df2 = new DecimalFormat("######0.0"); 
		try {
			AbstractClassifier classifier = new BMAOfNBWithGaussian();
			//"./data/UCI/sick.arff"
			
			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			String[] settings={"-H 0.1","-H 0.6","-H 0.7","-H 0.8","-H 0.9","-H 1","-H 1.1","-H 1.2","-H 1.3","-H 1.4","-H 1.5","-H 1.6","-H 1.7","-H 1.8","-H 1.9","-H 2"};
			HyperParameterLinearPlot plot=new HyperParameterLinearPlot("./data/UCI/haberman.arff",classifier,settings);
			for(int i=0;i<results.length;i++)
			{
				double x=results[i];
				System.out.print("("+df2.format((i+1)*0.1)+","+df.format(x*100)+") ");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
