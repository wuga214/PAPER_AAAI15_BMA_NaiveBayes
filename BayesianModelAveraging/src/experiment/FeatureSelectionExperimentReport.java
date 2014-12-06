package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.text.StrBuilder;


import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class FeatureSelectionExperimentReport extends AbstractExperimentReport {

	private Map<String, List<Double>> featureSelection2MetricMean;
	private Map<String, List<Double>> featureSelection2MetricStd;
	private String classifier;
	private String problem;
	private int maxNumFeatures;
	private String metricName;
	private int folds;

	@Override
	public void saveInFile(String path) {
		String fileName = Joiner.on("_").skipNulls().join(this.problem.replaceAll("[^a-zA-Z0-9\\s]", ""), this.classifier, this.metricName, ".m" );
		Path file = Paths.get(path,fileName);

		try(PrintWriter out = new PrintWriter(file.toFile())){

			out.print(this.saveString());			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	@Override
	public String saveString() {

		StrBuilder sb = new StrBuilder();
		sb.appendln(Joiner.on(",").skipNulls().join(this.problem.replaceAll("[^a-zA-Z0-9\\s]", ""), this.classifier, this.metricName,this.maxNumFeatures));

		for (String alg : featureSelection2MetricMean.keySet()) {
			sb.appendln(alg);
			
			if(featureSelection2MetricMean.get(alg) != null){
				sb.append("Mean,");
				sb.appendln(Joiner.on(",").skipNulls().join(featureSelection2MetricMean.get(alg)));
			}
			if(featureSelection2MetricStd.get(alg)!= null){
				sb.append("STD,");
				sb.appendln(Joiner.on(",").skipNulls().join(featureSelection2MetricStd.get(alg)));
			}
		}
		return sb.toString();
	}

	@Override
	public void plot(String path) {

		String fileName = Joiner.on("_").skipNulls().join(this.problem.replaceAll("[^a-zA-Z0-9\\s]", ""), this.classifier, this.metricName, ".m" );
		Path file = Paths.get(path,fileName);

		try(PrintWriter out = new PrintWriter(file.toFile())){

			out.print(this.plotString());			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	@Override
	public String plotString() {

		StrBuilder sb = new StrBuilder();

		Iterator<String> plotSymbols =   Iterables.cycle(
				"'-yo'","'-m+'","'-c*'","'-rx'","'-gs'","'-bd'","'-kp'",
				"'-y.'","'-m^'","'-cv'","'-r>'","'-g<'","'-bh'","'-ko'",
				"'-y+'","'-m*'","'-cx'","'-rs'","'-gd'","'-bp'","'-k.'",
				"'-y^'","'-mv'","'-c>'","'-r<'","'-gh'","'-bo'","'-k+'",
				"'-y*'","'-mx'","'-cs'","'-rd'","'-gp'","'-b.'","'-k^'",
				"'-yv'","'-m>'","'-c<'","'-rh'","'-go'","'-b+'","'-k*'").iterator();



		sb.appendln("%simulation Plot");

		sb.appendln("%open plot");
		sb.appendln("figure, hold on;");
		sb.appendln("");

		sb.appendln("%Simulation results");
		sb.appendln("n_features = [1: "+ this.maxNumFeatures + "];");
		sb.appendln("");

		List<String> legend = new ArrayList<String>();
		for (String alg : this.featureSelection2MetricMean.keySet()) {
			if(featureSelection2MetricMean.get(alg) != null){
				String metric = "metric_" + alg;
				String error = "err_" + alg;
				legend.add("'" + alg.replaceAll("_", " ") + "'");
				String symbol = plotSymbols.next();

				sb.appendln(metric + " = ["+ Joiner.on(", ").skipNulls().join(featureSelection2MetricMean.get(alg)) + "];");
				sb.appendln(error + " = 2.262 .* (1/"+Math.sqrt(folds)+") .* ["+ Joiner.on(", ").skipNulls().join(featureSelection2MetricStd.get(alg)) + "];");
				sb.appendln("errorbar(n_features, "+ metric + "," + error + ","+ symbol +")");
				sb.appendln("");
			}
		}

		sb.appendln("%plots settings");
		sb.appendln("title('"+ Joiner.on(" ").skipNulls().join(this.problem, this.classifier, this.metricName ) +"');");
		sb.appendln("xlabel('number of features');");
		sb.appendln("ylabel('"+ this.metricName +"');");
		sb.appendln("legend(" + Joiner.on(",").skipNulls().join(legend) + ",'Location', 'SouthOutside');");
		sb.appendln("saveas(gcf, '" + Joiner.on("_").skipNulls().join(this.problem, this.classifier, this.metricName ) + "' , 'pdf')");
		
		return sb.toString();
	}


	public FeatureSelectionExperimentReport() {
		super();
	}

	public FeatureSelectionExperimentReport(
			Map<String, List<Double>> featureSelection2metric, Map<String, List<Double>> featureSelection2metricStd,
			String classifier, String problem, int maxNumFeatures,int folds,
			String metricName) {
		super();
		this.featureSelection2MetricMean = featureSelection2metric;
		this.featureSelection2MetricStd = featureSelection2metricStd;
		this.classifier = classifier;
		this.problem = problem;
		this.maxNumFeatures = maxNumFeatures;
		this.metricName = metricName;
		this.folds = folds;
	}

	public FeatureSelectionExperimentReport(String classifierName, String problemName,
			int maxNumFeatures, int folds,String metricName) {
		this.classifier = classifierName;
		this.problem = problemName;
		this.maxNumFeatures = maxNumFeatures;
		this.featureSelection2MetricMean = new HashMap<String, List<Double>>();
		this.featureSelection2MetricStd = new HashMap<String, List<Double>>();
		this.metricName = metricName;
		this.folds = folds;
	}

	public Map<String, List<Double>> getFeatureSelection2metric() {
		return featureSelection2MetricMean;
	}

	public void setFeatureSelection2metric(Map<String, List<Double>> featureSelection2metric) {
		this.featureSelection2MetricMean = featureSelection2metric;
	}

	public String getClassifier() {
		return classifier;
	}

	public void setClassifier(String classifier) {
		this.classifier = classifier;
	}

	public String getProblem() {
		return problem;
	}

	public void setProblem(String problem) {
		this.problem = problem;
	}

	public int getMaxNumFeatures() {
		return maxNumFeatures;
	}

	public void setMaxNumFeatures(int maxNumFeatures) {
		this.maxNumFeatures = maxNumFeatures;
	}

	public String getMetricName() {
		return metricName;
	}

	public void setMetricName(String metricName) {
		this.metricName = metricName;
	}

	public Map<String, List<Double>> getFeatureSelection2MetricStd() {
		return featureSelection2MetricStd;
	}

	public void setFeatureSelection2MetricStd(
			Map<String, List<Double>> featureSelection2MetricStd) {
		this.featureSelection2MetricStd = featureSelection2MetricStd;
	}

	public void metricStdAddValue(String alg, double value){
		if(!this.featureSelection2MetricStd.containsKey(alg)){
			this.featureSelection2MetricStd.put(alg, new ArrayList<Double>());
		}
		this.featureSelection2MetricStd.get(alg).add(value);
	}
	public void metricMeanAddValue(String alg, double value){
		if(!this.featureSelection2MetricMean.containsKey(alg)){
			this.featureSelection2MetricMean.put(alg, new ArrayList<Double>());
		}
		this.featureSelection2MetricMean.get(alg).add(value);
	}

	public int getFolds() {
		return folds;
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}


}
