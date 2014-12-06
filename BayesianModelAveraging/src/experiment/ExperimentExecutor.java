package experiment;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.TrueFileFilter;

import problems.ClassificationProblem;
import utils.IConstants;

public class ExperimentExecutor {

	private static ExperimentExecutor instance;

	public static synchronized ExperimentExecutor getInstance() {
		if (instance == null) {
			instance = new ExperimentExecutor();
		}
		return instance;
	}

	private ExperimentExecutor(){}


	//execute a command per arrf problem in a jar data set
	public List<AbstractExperimentReport> executeCommandInJAR(IExperimentCommand cmd, String jardataSetPath){

		List<AbstractExperimentReport> results = new ArrayList<AbstractExperimentReport>();

		Path jarfile = Paths.get(jardataSetPath);
		try {
			JarFile file = new JarFile(jarfile.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();

				List<AbstractExperimentReport> partialResult = this.executeExperimentJAR(cmd, file, entry);
				if(partialResult != null && partialResult.size() > 0)
					results.addAll(partialResult);

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return results;
	}

	//execute a command per arrf problem in the files data set
	public List<AbstractExperimentReport> executeCommandInFiles(IExperimentCommand cmd, String dataSetSourcePath) {

		List<AbstractExperimentReport> results = new ArrayList<AbstractExperimentReport>();
		File dir = new File(dataSetSourcePath);
		if(dir.isDirectory()){
			Collection<File> files = FileUtils.listFiles(dir, TrueFileFilter.INSTANCE, TrueFileFilter.INSTANCE);
			for (File file: files) {
				List<AbstractExperimentReport> partialResult = this.executeExperimentFile(cmd, file.getPath());
				if(partialResult != null && partialResult.size() > 0)
					results.addAll(partialResult);
			}
		}

		return results;

	}
	//execute a command in a arff file
	public AbstractExperimentReport executeCommandInFile(IExperimentCommand cmd, String filePath) {

		List<AbstractExperimentReport> results = this.executeExperimentFile(cmd, filePath);		
		return results != null && results.size() > 0 ? results.get(0) : null;
	}

	//private menbers//*************8

	//execute a experiment with problems in files
	private List<AbstractExperimentReport> executeExperimentFile(IExperimentCommand iecmd,  String filePath) {

		List<AbstractExperimentReport> results = new ArrayList<AbstractExperimentReport>();
		try {
			File file = new File(filePath);
			if(file != null && file.exists()){
				String ext = com.google.common.io.Files.getFileExtension(file.getPath());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					ClassificationProblem cp = new ClassificationProblem(file.getPath());
					List<AbstractExperimentReport> report = iecmd.execute(cp);
					if(report != null && report.size() > 0)
						results.addAll(report);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return results;
	}

	//execute a experiment with problems in jar
	private List<AbstractExperimentReport> executeExperimentJAR(IExperimentCommand iecmd,  JarFile file, JarEntry entry){

		List<AbstractExperimentReport> results = new ArrayList<AbstractExperimentReport>();
		try {
			if(entry != null){
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry));
					ClassificationProblem cp = new ClassificationProblem(entry.getName(),inpr);
					List<AbstractExperimentReport> report = iecmd.execute(cp);
					if(report != null && report.size() > 0)
						results.addAll(report);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return results;
	}

}
