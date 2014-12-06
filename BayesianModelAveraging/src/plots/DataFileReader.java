package plots;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;

public class DataFileReader {

	private static double[] result;
	private static double[] strToArray(String str) {

		String[] strAry = str.split(",");
		double[] ary = new double[strAry.length];
		for(int i = 0; i < strAry.length; i++){
			ary[i] = Double.parseDouble(strAry[i]);
		}
		return ary;
	}
	
	public static void doPrint()
	{
		DecimalFormat df = new DecimalFormat("######0.0000");  
		for(int i=0;i<result.length;i++)
		{
			if(i==0){
			System.out.print("("+1+","+df.format(result[i])+") ");
			}
			else
			{
			System.out.print("("+(i*10)+","+df.format(result[i])+") ");
			}
		}	
	}
	
    public static void readFileByLines(String fileName) {
        File file = new File(fileName);
        BufferedReader reader = null;
        try {
            System.out.println("read as line");
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            int line = 1;
            // finish when read null
            while ((tempString = reader.readLine()) != null) {
                // display line number
            	System.out.println("line " + line + ": " + tempString);
            	double[] temp=strToArray(tempString);
               if(result!=null){
           		for(int i=0;i<result.length;i++)
           			{
        			result[i]+=temp[i];
           			}
               	}
               else{result=temp;}
                line++;
            	}
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
        for(int i=0;i<result.length;i++)
		{
		result[i]=result[i]/45;
		}
        doPrint();
    }
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		readFileByLines("./SVM.txt");
	}

}
