package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class BinaryClassificationFileMaker {
	public static void removeLineFromFile(String file, String lineToRemove) {

		try {

		  File inFile = new File(file);

		  if (!inFile.isFile()) {
		    System.out.println("Parameter is not an existing file");
		    return;
		  }

		  //Construct the new file that will later be renamed to the original filename.
		  File tempFile = new File(inFile.getAbsolutePath() + ".tmp");

		  BufferedReader br = new BufferedReader(new FileReader(file));
		  PrintWriter pw = new PrintWriter(new FileWriter(tempFile));

		  String line = null;

		  //Read from the original file and write to the new
		  //unless content matches data to be removed.
		  while ((line = br.readLine()) != null) {

		    if (!line.trim().contains(lineToRemove)) {

		      pw.println(line);
		      pw.flush();
		    }
		  }
		  pw.close();
		  br.close();

		  //Delete the original file
		  if (!inFile.delete()) {
		    System.out.println("Could not delete file");
		    return;
		  }

		  //Rename the new file to the filename the original file had.
		  if (!tempFile.renameTo(inFile))
		    System.out.println("Could not rename file");

		}
		catch (FileNotFoundException ex) {
		  ex.printStackTrace();
		}
		catch (IOException ex) {
		  ex.printStackTrace();
		}
	}
		
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		removeLineFromFile("./data/NEWS/20news2.arff", "9}");
	}

}
