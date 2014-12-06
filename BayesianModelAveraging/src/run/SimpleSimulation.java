package run;

import java.util.List;

import ExperimentCommands.SimpleCommandExample;
import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;

public class SimpleSimulation {
	public static void main(String[] args) {
		double start = System.currentTimeMillis();
		String path = "./data";
		String jarPath = "./data/datasets-UCI.jar";
		IExperimentCommand cmd = new SimpleCommandExample();
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, path);
		//List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInJAR(cmd, jarPath);
		AbstractExperimentReport.saveAll(result, "./results/backgroudtest.csv");
		System.out.println("elapsed time: " + (System.currentTimeMillis() - start));
	}
}
