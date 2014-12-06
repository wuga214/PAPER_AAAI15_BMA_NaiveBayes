package plots;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import classifiers.ClassifierFactory;
import classifiers.ELinearClassifier;

import com.google.common.collect.Sets;

import evaluation.WekaEvaluationWrapper;
import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;
/**
 * 
 * @author wuga
 * Classifier vs. #Feature plots generator 
 * plot one single line with respect to classifier
 * run multiple time for all classifer to get complete linear plots
 * parameter setting need to modify for different classifier
 * please see comments below
 * C and B are for SVM and LR
 * Q and H are for BMA
 * there is no setting for Naive Bayes
 */
public class FALinearPlot {
	public static double[][] results;
	public double[] result;
	public ArrayList<Integer> featureindex;
	public ArrayList<Integer> randomizedfeatureindex;
	public int m_numberOfAttribute=1000;
	public int iterate;
	
	public FALinearPlot(ClassificationProblem cp,AbstractClassifier classifier) throws Exception{
		iterate=45;
		results=new double[iterate][m_numberOfAttribute/10+1];
		result=new double[m_numberOfAttribute/10+1];
		String[] Options=new String[2];
		ClassificationProblem x = new ClassificationProblem("./data/NEWS/news.arff");
		WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);
		for(int i=0;i<iterate;i++)
		{
			Remove attributeFilter=null;
			attributeFilter = new Remove();
			randomizedfeatureindex=new ArrayList<Integer>();
			GenerateList(m_numberOfAttribute-1);
			randomizedPermutation();
			for(int j=1;j<=m_numberOfAttribute+1;j=j+10)
			{
			    attributeFilter.setAttributeIndicesArray(pickSubsetFeatures(j-1,cp.getNumAttributes()));
			    attributeFilter.setInvertSelection(true);
			    attributeFilter.setInputFormat(cp.getData());
				Instances newData=Filter.useFilter(cp.getData(), attributeFilter);
				x.setData(newData);
				AbstractClassifier copiedClassifier=(AbstractClassifier) AbstractClassifier.makeCopy(classifier);
				HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			//	paramLR.put("-Q",Sets.newHashSet("-Q"));
//				paramLR.put("-H",Sets.newHashSet("-H 0.9"));
			//	paramLR.put("-H",Sets.newHashSet("-H 0.8","-H 0.9","-H 1","-H 1.1"));
			//	paramLR.put("-C",Sets.newHashSet("-C 1","-C 10","-C 100"));
			//	paramLR.put("-B",Sets.newHashSet("-B 1"));
			//	eval.crossValidateModelTuneInAccuraccy(copiedClassifier, paramLR,x, 10, 10, null);
				eval.CrossValidation(copiedClassifier, null,x, 10, 10, paramLR,null);
				results[i][j/10]=eval.accuracy();
				if(j==1){j=j-1;}
			}
		}
		for(int i=0;i<result.length;i++)
		{
			result[i]=0;
			for(int j=0;j<iterate;j++)
			{
			result[i]+=results[j][i];
			}
			result[i]=result[i]/iterate;
		}
	}
	
	public void doPrint()
	{
		DecimalFormat df = new DecimalFormat("######0.0000");  
		for(int i=0;i<result.length;i++)
		{
			if(i==0){
			System.out.print("("+1+","+df.format(result[i])+") ");
			}
			else
			{
			System.out.print("("+(i*10)+","+df.format(result[i])+") ");
			}
		}	
	}
	
	public int[] randomPermutation(int numberoffeatures,int randSeed, int requiredFeatures)
	{
		int[] seed=new int[numberoffeatures-1];
		for(int i=0;i<numberoffeatures-1;i++)
		{
			seed[i]=i;
		}
	    int len=seed.length;  
	    int[] result= new int[requiredFeatures+1];   
	    Random random = new Random(randSeed);   
	    for (int i = 0; i < requiredFeatures; i++) {   
	      int r = random.nextInt(len - i);   
	      result[i] = seed[r];   
	      seed[r] = seed[len - 1 - i];   
	    }
	    result[requiredFeatures]=numberoffeatures-1;
	    return result;
	}
	
     
	public void randomizedPermutation(){
		Random rand=new Random();
		if(featureindex.size()!=0)
		{
			int selectedindex=rand.nextInt(featureindex.size());
			randomizedfeatureindex.add(featureindex.get(selectedindex));
			featureindex.remove(selectedindex);
			randomizedPermutation();			
		}
		else
		{
			for(int i=0;i<randomizedfeatureindex.size();i++)
			{
				System.out.print(randomizedfeatureindex.get(i)+" ");
			}
			System.out.println();
			GenerateList(m_numberOfAttribute);
		}
		
	}
	
	public int[] pickSubsetFeatures(int num,int indexoflabel){
		int[] featurelist=new int[num+1];
		for(int i=0;i<num;i++)
		{
			featurelist[i]=randomizedfeatureindex.get(i);
		}
		featurelist[num]=indexoflabel-1;
		return featurelist;
	}
	
	public void GenerateList(int numberoffeatures){
		ArrayList<Integer> numList=new ArrayList<Integer>();
		for(int i=0;i<numberoffeatures;i++){
			numList.add(i+1);
		}
		featureindex=numList;
	}

	private static String aryToString(double[] ary){
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < ary.length; i++){
			sb.append(ary[i]).append(",");
		}
		
		sb.setCharAt(sb.length() -1, '\n');
		
		return sb.toString();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try{
		ClassificationProblem cp = new ClassificationProblem("./data/NEWS/news.arff");
		ELinearClassifier c=ELinearClassifier.C_NAIVE_BAYES;
		ClassifierFactory factory=new ClassifierFactory();
		AbstractClassifier classifier=factory.createClassifier(c);
		FALinearPlot fa=new FALinearPlot(cp,classifier);
		//fa.doPrint();
		File file = new File("./NB.txt");
		if (!file.exists()) {
		    file.createNewFile();
		   }
		FileWriter fw = new FileWriter(file.getAbsoluteFile(),true);
		BufferedWriter bw = new BufferedWriter(fw);
		for(int i=0;i<results.length;i++){
		bw.write(aryToString(results[i]));
		}
		bw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}

}
