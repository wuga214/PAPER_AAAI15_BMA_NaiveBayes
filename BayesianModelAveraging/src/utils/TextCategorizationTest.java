package utils;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.*;
import weka.core.converters.*;
import weka.classifiers.trees.*;
import weka.filters.*;
import weka.filters.supervised.attribute.*;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.util.Random;

/**
 * Example class that converts HTML files stored in a directory structure into 
 * and ARFF file using the TextDirectoryLoader converter. It then applies the
 * StringToWordVector to the data and feeds a J48 classifier with it.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class TextCategorizationTest {

  /**
   * Expects the first parameter to point to the directory with the text files.
   * In that directory, each sub-directory represents a class and the text
   * files in these sub-directories will be labeled as such.
   *
   * @param args        the commandline arguments
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // convert the directory into a dataset
    TextDirectoryLoader loader = new TextDirectoryLoader();
    loader.setDirectory(new File("./data/textData"));
    Instances dataRaw = loader.getDataSet();
    //System.out.println("\n\nImported data:\n\n" + dataRaw);

    // apply the StringToWordVector
    // (see the source code of setOptions(String[]) method of the filter
    // if you want to know which command-line option corresponds to which
    // bean property)
    StringToWordVector filter = new StringToWordVector();
    filter.setInputFormat(dataRaw);
    Instances dataFiltered = Filter.useFilter(dataRaw, filter);
    weka.filters.supervised.attribute.AttributeSelection filter2 = new weka.filters.supervised.attribute.AttributeSelection();  // package weka.filters.supervised.attribute!
//    CfsSubsetEval eval = new CfsSubsetEval();
    InfoGainAttributeEval eval=new InfoGainAttributeEval();
//    GreedyStepwise search = new GreedyStepwise();
    Ranker searchMethod = new Ranker();
    searchMethod.setNumToSelect(100);
    //searchMethod.setSearchBackwards(true);
    filter2.setEvaluator(eval);
    filter2.setSearch(searchMethod);
    filter2.setInputFormat(dataFiltered);
    //generate new data
    Instances newData = Filter.useFilter(dataFiltered, filter2);
    		newData.randomize(new Random());
    		System.out.println("number of data:"+newData.numInstances());
    		System.out.println("number of attributes:"+newData.numAttributes());
    		 ArffSaver saver = new ArffSaver();
    		 saver.setInstances(newData);
    		 saver.setFile(new File("./data/news3000.arff"));
    		 saver.setDestination(new File("./data/news3000.arff"));   // **not** necessary in 3.5.4 and later
    		 saver.writeBatch();

  }
}