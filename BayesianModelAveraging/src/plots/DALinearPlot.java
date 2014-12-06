package plots;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import classifiers.ClassifierFactory;
import classifiers.ELinearClassifier;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.unsupervised.attribute.RandomSubset;

/**
 * 
 * @author wuga
 * Classifier vs. #of Data plots generator 
 * plot one single line with respect to classifier
 * run multiple time for all classifer to get complete linear plots
 * parameter setting need to modify for different classifier
 * please see comments below
 * C and B are for SVM and LR
 * Q and H are for BMA
 * there is no setting for Naive Bayes
 */
public class DALinearPlot {
	public double[][] results;
	public double[] result;
	public int m_numberOfData=100;
	public int iterator=100;
	private Evaluation wekaEvaluation ;
	
	public DALinearPlot(ClassificationProblem cp,AbstractClassifier classifier) throws Exception{
		RandomSubset filter=new RandomSubset();
		Resample filter2=new Resample();
		results=new double[iterator][m_numberOfData];
		result=new double[m_numberOfData];
		String[] Options=new String[4];
		String[] Options2=new String[2];
		ClassificationProblem x = new ClassificationProblem("./data/NEWS/news3000.arff");
	//	WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
		Random rand=new Random();
		for(int i=0;i<iterator;i++)
		{
			System.out.println("iteration:"+(i+1));
			/*	Options[0]="-S";
			Options[1]= Integer.toString(rand.nextInt());
			Options[2]="-N";
			Options[3]="100";
			filter.setOptions(Options);
			filter.setInputFormat(cp.getData());
			Instances newData=Filter.useFilter(cp.getData(), filter);*/
		/*	Options2[0]="-S";
			Options2[1]= Integer.toString(rand.nextInt());
			Options2[2]="-Z";
			Options2[3]= "100";
			Options2[4]="-B";
			Options2[5]="1";
			Options2[6]="-no-replacement";
			filter2.setOptions(Options2);
			filter2.setInputFormat(cp.getData());
			Instances filteredData=Filter.useFilter(cp.getData(), filter2);*/
			Instances TVdata=cp.getData().trainCV(5, 1, rand);
			Instances testing=cp.getData().testCV(5, 1);
			for(int j=1;j<=m_numberOfData;j++)
			{
				
			//	Options2[0]="-S";
			//	Options2[1]= Integer.toString(rand.nextInt());
				Options2[0]="-Z";
				Options2[1]= Integer.toString(j);
			//	Options2[4]="-B";
			//	Options2[5]="1";
			//	Options2[6]="-no-replacement";
				filter2.setOptions(Options2);
				filter2.setInputFormat(TVdata);
				Instances newData2=Filter.useFilter(TVdata, filter2);
				Instances training=newData2.trainCV(5, 1);
				Instances validating=newData2.testCV(5, 1);
			//	System.out.println(newData2.numInstances());
				AbstractClassifier copiedClassifier=(AbstractClassifier) AbstractClassifier.makeCopy(classifier);
				HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
				paramLR.put("-Q",Sets.newHashSet("-Q"));
//				paramLR.put("-H",Sets.newHashSet("-H 0.9"));
				paramLR.put("-H",Sets.newHashSet("-H 0.8","-H 0.9","-H 1","-H 1.1"));
			//	paramLR.put("-C",Sets.newHashSet("-C 1","-C 10","-C 100"));
			//	paramLR.put("-B",Sets.newHashSet("-B 1"));
			//	paramLR.put("-C",Sets.newHashSet("-C 1.0", "-C 10", "-C 100"));
			//	paramLR.put("-B",Sets.newHashSet("-B 1.0", "-B 10", "-B 100"));
			//	eval.crossValidateModelTuneInAccuraccy(copiedClassifier, paramLR,x, 10, 10, null);
			//	eval.CrossValidation(copiedClassifier, null,x, 10, 10, paramLR,null);
				String setting=tuneClassifier( copiedClassifier,paramLR, null, training, validating,newData2);
				AbstractClassifier copiedClassifier2=(AbstractClassifier) AbstractClassifier.makeCopy(classifier);
				copiedClassifier2.setOptions(Utils.splitOptions(setting));
				copiedClassifier2.buildClassifier(newData2);
			this.evaluateModel(copiedClassifier2, testing,newData2);
				results[i][j-1]=wekaEvaluation.pctCorrect();
			}
		}
		for(int i=0;i<result.length;i++)
		{
			result[i]=0;
			for(int j=0;j<iterator;j++)
			{
			result[i]+=results[j][i];
			}
			result[i]=result[i]/iterator;
		}
	}
	
	public void doPrint()
	{
		DecimalFormat df = new DecimalFormat("######0.0000");  
		for(int i=0;i<result.length;i++)
		{

			System.out.print("("+(i+1)+","+df.format(result[i])+") ");

		}	
	}
	
	private String tuneClassifier(AbstractClassifier classifier,
			Map<String, Set<String>> params, EClassificationMetric metric,
			Instances trainData, Instances validationData,Instances dataset) throws Exception {

		//tune the model parameters
		String optimumSetting = "";
		if(params != null && !params.isEmpty()){

			List<String> modelSettings = Util.generateModels(Lists.newArrayList(params.values()));
			double maxMetric = Double.MIN_VALUE;
			for (String setting : modelSettings) {
				//set parameter, train and evaluate
				classifier.setOptions(Utils.splitOptions(setting));
				classifier.buildClassifier(trainData);
				this.evaluateModel(classifier, validationData,dataset);
				double accuracy=wekaEvaluation.pctCorrect();

				if(accuracy  > maxMetric){
					maxMetric = accuracy;
					optimumSetting = setting;
				}
			}

		}
		return optimumSetting;
	}
	
	public void evaluateModel(AbstractClassifier c, Instances test,Instances dataset) throws Exception {
		this.wekaEvaluation = new Evaluation(dataset);
		this.wekaEvaluation.evaluateModel(c, test);
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try{
		ClassificationProblem cp = new ClassificationProblem("./data/NEWS/news3000.arff");
		ELinearClassifier c=ELinearClassifier.C_BMA;
		ClassifierFactory factory=new ClassifierFactory();
		AbstractClassifier classifier=factory.createClassifier(c);
		DALinearPlot fa=new DALinearPlot(cp,classifier);
		fa.doPrint();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}

}
