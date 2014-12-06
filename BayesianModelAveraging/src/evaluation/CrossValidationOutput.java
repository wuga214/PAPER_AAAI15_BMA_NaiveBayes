package evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.StatUtils;



public class CrossValidationOutput {

	private List<FoldResult> foldsResults;
	private long seed;
	private int folds;
	private static final DecimalFormat  formatter = new DecimalFormat ("#0.000");

	public CrossValidationOutput(long seed,
			int folds) {

		super();
		this.foldsResults = new ArrayList<FoldResult>();
		this.seed = seed;
		this.folds = folds;
	}
	
	public CrossValidationOutput(
			int folds) {
		this(0,folds);
	}
	
	public double metricMean(EClassificationMetric metric){
		double[] metricVector = new double[this.foldsResults.size()]; 
		for (int idx = 0 ; idx < metricVector.length ; idx++) {
			metricVector[idx] = this.foldsResults.get(idx).getMetricReported(metric);
		}
		return StatUtils.mean(metricVector);
	}

	public double metricSTD(EClassificationMetric metric){
		double[] metricVector = new double[this.foldsResults.size()]; 
		for (int idx = 0 ; idx < metricVector.length ; idx++) {
			metricVector[idx] = this.foldsResults.get(idx).getMetricReported(metric);
		}
		return Math.sqrt(StatUtils.variance(metricVector));
	}
	
	//getters and setters
	public List<FoldResult> getFoldsResults() {
		return foldsResults;
	}
	public FoldResult getFoldResult(int index) {
		return foldsResults.get(index);
	}

	public void addFoldResult(FoldResult foldsResult) {
		this.foldsResults.add(foldsResult);
	}

	public long getSeed() {
		return seed;
	}

	public void setSeed(long seed) {
		this.seed = seed;
	}

	public int getFolds() {
		return folds;
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}


	@Override
	public String toString() {
		return "CrossValidationOutput [accuracyMean()=" + formatter.format(metricMean(EClassificationMetric.ACCURACY))
				+ ", precisionMean()=" + formatter.format(metricMean(EClassificationMetric.PRECISION)) + ", recallMean()="
				+ formatter.format(metricMean(EClassificationMetric.RECALL)) + ", fmeasureMean()=" + formatter.format(metricMean(EClassificationMetric.FSCORE))
				+ ", accuracyStd()=" + formatter.format(metricSTD(EClassificationMetric.ACCURACY)) + ", precisionStd()="
				+ formatter.format(metricSTD(EClassificationMetric.PRECISION)) + ", recallyStd()=" + formatter.format(metricSTD(EClassificationMetric.RECALL))
				+ ", fmeasureStd()=" + formatter.format(metricSTD(EClassificationMetric.FSCORE)) + ", seed=" + seed
				+ ", folds=" + folds + "]";
	}

}
