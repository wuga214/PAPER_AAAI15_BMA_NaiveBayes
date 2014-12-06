package utils;

public class NumericalUtils {
/* for a,b given log versions log_a, log_b returns log(a + b);
* also known as log-sum-exp
*
* NOTE: code needs to be modified further if INF or -INF values
*       are expected.
*/
	public static double LogSum(double log_a, double log_b) {
		double max = Math.max(log_a, log_b);
		double test1=Math.exp(log_a - max);
		double test2=Math.exp(log_b - max);
		double sum = test1 + test2;
		return Math.log(sum) + max;
	//	if(log_a>log_b){return log_a;}
	//	else{return log_b;}
	}
	public static void main(String[] args) {
// Test LogSum
		double a = -40;
		double b = -35;
		double a_plus_b = a + b;
		//double log_a = Math.log(a);
		//double log_b = Math.log(b);
		double log_a = -0;
		double log_b = -1000;
		double log_a_plus_b = LogSum(log_a, log_b);
		System.out.println("LogSum: "+ LogSum(log_a, log_b));
		//System.out.println("LogSum: " + a_plus_b + " == " + Math.exp(log_a_plus_b));
	}
}