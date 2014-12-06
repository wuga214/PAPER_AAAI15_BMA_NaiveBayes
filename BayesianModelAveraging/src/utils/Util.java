package utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.text.WordUtils;
import org.apache.commons.lang3.time.DateUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import problems.ClassificationProblem;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import experiment.AbstractExperimentReport;
import experiment.FeatureSelectionExperimentReport;


public class Util {


	public static void  writeToXml(List<AbstractExperimentReport> reports, String xmlResultPath){
		try {

			XPath xpath = XPathFactory.newInstance().newXPath();

			DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
			Document doc = docBuilder.newDocument();
			Element root = doc.createElement("Simulation");
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			root.setAttribute("date", dateFormat.format(date));
			doc.appendChild(root);

			for (AbstractExperimentReport abr : reports) {
				FeatureSelectionExperimentReport report = (FeatureSelectionExperimentReport)abr;

				Element dataSet = (Element) xpath.evaluate("//DataSet[@name='"+ report.getProblem() + "']", doc, XPathConstants.NODE);  
				if(dataSet == null){
					// root elements
					dataSet = doc.createElement("DataSet");
					dataSet.setAttribute("name", report.getProblem());
					dataSet.setAttribute("nfeatures", Integer.toString(report.getMaxNumFeatures()));
					root.appendChild(dataSet);
				}


				// classifier elements
				Element classifier = (Element) xpath.evaluate("//DataSet[@name='"+ report.getProblem() + "']"+"/Classifier[@name='"+ report.getClassifier() + "']", doc, XPathConstants.NODE);  
				if(classifier == null)
				{
					classifier = doc.createElement("Classifier");
					classifier.setAttribute("name", report.getClassifier());
					dataSet.appendChild(classifier);
				}


				//metrics elements
				Element metric = (Element) xpath.evaluate("//DataSet[@name='"+ report.getProblem() + "']"+"/Classifier[@name='"+ report.getClassifier() + "']" + "/Metric[@name='"+ report.getMetricName() + "']", doc, XPathConstants.NODE);
				if(metric == null){
					metric = doc.createElement("Metric");
					metric.setAttribute("name", report.getMetricName());
					classifier.appendChild(metric);
				}

				//algs elements
				Element algorithms = doc.createElement("Algorithms");
				metric.appendChild(algorithms);


				for (String algName : report.getFeatureSelection2metric().keySet()) {
					Element alg = doc.createElement("Algorithm");
					alg.setAttribute("name", algName);

					Element mean = doc.createElement("Mean");
					mean.appendChild(doc.createTextNode("["+ Joiner.on(", ").skipNulls().join(report.getFeatureSelection2metric().get(algName)) + "];"));

					Element Error = doc.createElement("Error");
					Error.setAttribute("folds", Integer.toString(report.getFolds()));
					Error.appendChild(doc.createTextNode("[" + Joiner.on(", ").skipNulls().join(report.getFeatureSelection2MetricStd().get(algName)) + "];"));

					
					algorithms.appendChild(alg);
					alg.appendChild(mean);
					alg.appendChild(Error);

				}
			}

			// write the content into xml file
			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			transformer.setOutputProperty(OutputKeys.INDENT, "yes");
			transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "5");
			DOMSource source = new DOMSource(doc);
			StreamResult result = new StreamResult(new File(xmlResultPath));

			// Output to console for testing
			//StreamResult result = new StreamResult(System.out);

			transformer.transform(source, result);


			System.out.println("File saved!");

		} catch (ParserConfigurationException pce) {
			pce.printStackTrace();
		} catch (TransformerException tfe) {
			tfe.printStackTrace();
		} catch (XPathExpressionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String cleanLabels(String text){
		return StringUtils.capitalize(text.replaceAll("[^a-zA-Z0-9\\s]", " ").toLowerCase());
	}

	public static List<ClassificationProblem> readAllFilesARFFFromDirectory(String path){
		List<ClassificationProblem> datasets = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				String ext = com.google.common.io.Files.getFileExtension(file.toString());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION))
					datasets.add(new ClassificationProblem(file.toFile().getPath()));
			}
		} catch (IOException | DirectoryIteratorException x) {
			x.printStackTrace();
		}

		System.out.println(datasets.size() +" ARFF files were read");
		return datasets;
	}

	public static List<ClassificationProblem> readAllFilesARFFFromJar(String path){
		List<ClassificationProblem> dataSets = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try {
			JarFile file = new JarFile(dir.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry)) ;
					dataSets.add(new ClassificationProblem(entry.getName(),inpr));
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println(dataSets.size() +" ARFF files were read");
		return dataSets;
	}


	//generate all possibles models based on param
	public static List<String> generateModels(List<Set<String>> paramsList){
		List<String> modelsStringSetting = null;
		final Joiner joiner = Joiner.on(" ").skipNulls();
		Function<List<String>, String> buildModelSettingString = new Function<List<String>, String>() {
			public String apply(List<String> params) {
				return joiner.join(params);
			}
		};

		if(paramsList != null && paramsList.size() > 1){
			Set<List<String>> cartesianProd = Sets.cartesianProduct(paramsList);

			if(cartesianProd != null){
				modelsStringSetting = Lists.newArrayList( Iterables.transform(cartesianProd, buildModelSettingString));
			}
		}

		return modelsStringSetting;
	}

	public static Set<String> generateModelsStringSettings(String settingCode, double... values){

		Set<String> ret = new HashSet<String>();
		for (double d : values) {
			ret.add(settingCode + " " + d);
		}
		return ret;
	}

	public static Logger getFileLogger(String logName, String logPath){
		try {
			SimpleFormatter formatter = new SimpleFormatter();  
			FileHandler file = new FileHandler(logPath);
			file.setFormatter(formatter);

			Logger.getLogger(logName).addHandler(file);

		} catch (SecurityException | IOException e) {
			e.printStackTrace();
		}

		return Logger.getLogger(logName);
	}

	public static String[] transformStringArray(int[] idxs){

		if(idxs == null | idxs.length <= 0)
			return null;

		String[] s = new String[idxs.length];
		for (int i = 0; i < s.length; i++) {
			s[i] = Integer.toString(idxs[i]);
		}
		return s;
	}
}
