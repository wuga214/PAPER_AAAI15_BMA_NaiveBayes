package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public abstract class AbstractExperimentReport {
	
	public abstract void saveInFile(String path);
	public abstract void plot(String filePath);
	public abstract String saveString();
	public abstract String plotString();
	
	public static void saveAll(List<AbstractExperimentReport> exps, String filePath){
		if(exps != null && exps.size() > 0){
			Path file = Paths.get(filePath);
			try(PrintWriter out = new PrintWriter(file.toFile())){
				for (AbstractExperimentReport exp : exps) {
					if(exp != null)
						out.println(exp.saveString());
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static void plotAll(List<AbstractExperimentReport> exps, String filePath){
		if(exps != null && exps.size() > 0){
			Path file = Paths.get(filePath);
			try(PrintWriter out = new PrintWriter(file.toFile())){
				for (AbstractExperimentReport exp : exps) {
					if(exp != null)
						out.println(exp.plotString());
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
}
