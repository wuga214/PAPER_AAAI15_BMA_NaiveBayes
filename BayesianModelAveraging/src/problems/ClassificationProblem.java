package problems;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.io.FileUtils;

import utils.IConstants;
import utils.Util;
import weka.core.Instances;


/**
 * @author RFSC
 *class to represent a classification problem
 */
public class ClassificationProblem {

	//data structure from weka
	private Instances data;
	//file path of the arff file
	private String filePath;
	//true label
	private Integer trueLabelValue;
	
	
	//constructor
	public ClassificationProblem(String filePath) throws IOException{
		this(filePath, new FileReader(filePath));
		
		if(IConstants.getInstance().getDataSet2TrueLabels().containsKey(this.getName()))
			this.trueLabelValue = IConstants.getInstance().getDataSet2TrueLabels().get(this.getName());
	}

	//contructor
	public ClassificationProblem(String filepath,InputStreamReader ird) throws IOException{

		if(ird != null){
			//set the filepath of arff file
			this.setFilePath(filepath);
			try (BufferedReader br = new BufferedReader(ird))
			{
				this.data = new Instances(br);
				this.data.setClassIndex(data.numAttributes() - 1);

			} 
		}
	}


	//getters and setters
	public int getNumAttributes(){
		return data.numAttributes();
	}

	public int getNumFeatures(){
		return data.numAttributes();
	}

	public int getNumExamples(){
		return data.numInstances();
	}

	public String getName(){
		return data.relationName();
	}

	public Instances getData() {
		return data;
	}

	public void setData(Instances newData) {
		this.data = newData;
	}

	public String getFilePath() {
		return filePath;
	}

	public void setFilePath(String filePath) {
		this.filePath = Paths.get(filePath).toString();;
	}

	//methods
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nname: " + this.getName());
		sb.append("\nfilepath: " + this.getFilePath());
		sb.append("\nNumber Of intances: " + this.getNumExamples());
		sb.append("\nNumber Of Attributes: " + this.getNumAttributes());
		sb.append("\nData:" + this.getData().toString());
		return sb.toString();
	}

	//test this class and the utils methods
	public static void main(String[] args) {
		//List<ClassificationProblem> problems = Util.readAllFilesARFFFromJar(IConstants.jAR_DATASETS_PATH);
		List<ClassificationProblem> problems = Util.readAllFilesARFFFromDirectory(IConstants.DATA_DIRECTORY_PATH);

		ClassificationProblem cp = problems.get(0);
		System.out.println(cp.data.classAttribute().toString());



	}

	public Integer getTrueLabelValue() {
		return trueLabelValue;
	}

	public void setTrueLabelValue(Integer trueLabelValue) {
		this.trueLabelValue = trueLabelValue;
	}



}
