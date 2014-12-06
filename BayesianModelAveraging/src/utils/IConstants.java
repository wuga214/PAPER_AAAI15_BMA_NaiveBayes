package utils;

import java.util.HashMap;

public class IConstants {

	private static IConstants instance = null;

	public static IConstants getInstance() {
		if(instance == null) {
			instance = new IConstants();
		}
		return instance;
	}
	
	public static final String ARFF_EXTENSION = "arff";
	public static final String DATA_DIRECTORY_PATH = "data";
	public static final String jAR_DATASETS_PATH = "data/datasets-UCI.jar";
	//set the true values labels
	private HashMap<String, Integer> dataSet2TrueLabels;

	protected IConstants(){
		if(dataSet2TrueLabels == null){
			//attention to choose the true labels
			dataSet2TrueLabels = new HashMap<String, Integer>();
			dataSet2TrueLabels.put("wisconsin-breast-cancer", 1);
			dataSet2TrueLabels.put("pima_diabetes", 1);
			dataSet2TrueLabels.put("heart-statlog", 1);
			dataSet2TrueLabels.put("spect", 1);
			dataSet2TrueLabels.put("vote", 1);
			dataSet2TrueLabels.put("newsgroup", 0);
		}
	}

	public HashMap<String, Integer> getDataSet2TrueLabels() {
		return dataSet2TrueLabels;
	}

}
