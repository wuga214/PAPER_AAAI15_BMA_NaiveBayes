package featureSelection;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.StartSetHandler;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.core.Range;

import com.google.common.primitives.Ints;

public class MyGreedySearch extends ASSearch implements StartSetHandler {

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** The number of attributes to select. -1 indicates that all attributes
	      are to be retained. Has precedence over m_threshold */
	protected int m_numToSelect = -1;

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** the best subset found */
	protected BitSet m_best_group;
	protected ASEvaluation m_ASEval;

	protected Instances m_Instances;

	/** holds the start set for the search as a Range */
	protected Range m_startRange;

	/** holds an array of starting attributes */
	protected int [] m_starting;

	/** Use a backwards search instead of a forwards one */
	protected boolean m_backward = false;

	public MyGreedySearch () {
		m_startRange = new Range();
		m_starting = null;
		resetOptions();
	}


	/**
	 * Searches the attribute subset space by forward selection.
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data)
			throws Exception {

		int i;
		double best_merit = -Double.MAX_VALUE;
		double temp_best,temp_merit;
		int temp_index=0;
		BitSet temp_group;

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}
		m_ASEval = ASEval;

		m_numAttribs = m_Instances.numAttributes();

		if (m_best_group == null) {
			m_best_group = new BitSet(m_numAttribs);
		}

		if (!(m_ASEval instanceof SubsetEvaluator)) {
			throw  new Exception(m_ASEval.getClass().getName() 
					+ " is not a " 
					+ "Subset evaluator!");
		}

		List<Integer> trackList = new ArrayList<Integer>();
		m_startRange.setUpper(m_numAttribs-1);
		if (!(getStartSet().equals(""))) {
			m_starting = m_startRange.getSelection();
		}
		
		m_classIndex = m_Instances.classIndex();
		

		SubsetEvaluator ASEvaluator = (SubsetEvaluator)m_ASEval;

		// If a starting subset has been supplied, then initialise the bitset
		if (m_starting != null) {
			for (i = 0; i < m_starting.length; i++) {
				if ((m_starting[i]) != m_classIndex) {
					m_best_group.set(m_starting[i]);
					trackList.add(m_starting[i]);
				}
			}
		} else {
			if (m_backward) {
				for (i = 0; i < m_numAttribs; i++) {
					if (i != m_classIndex) {
						m_best_group.set(i);
					}
				}
			}
		}

		// Evaluate the initial subset
		best_merit = ASEvaluator.evaluateSubset(m_best_group);

		// main search loop
		boolean done = false;
		boolean addone = false;
		boolean z;
		
		while (!done) {
			temp_group = (BitSet)m_best_group.clone();			
			temp_best = -Double.MAX_VALUE;

			if(m_backward){
				if(m_numToSelect > 0 && ((m_numAttribs - 1) - m_best_group.cardinality()) >= m_numToSelect)
					break;
			}else{
				if(m_numToSelect > 0 && m_best_group.cardinality() >= m_numToSelect)
					break;
			}

			done = true;
			addone = false;
			for (i=0;i<m_numAttribs;i++) {
				if (m_backward) {
					z = ((i != m_classIndex) && (temp_group.get(i)));
				} else {
					z = ((i != m_classIndex) && (!temp_group.get(i)));
				}
				if (z) {
					// set/unset the bit
					if (m_backward) {
						temp_group.clear(i);
					} else {
						temp_group.set(i);
					}
					temp_merit = ASEvaluator.evaluateSubset(temp_group);

					if (m_backward) {
						z = (temp_merit >= temp_best);
					} else {
						z = (temp_merit > temp_best);
					}

					if (z) {
						temp_best = temp_merit;
						temp_index = i;
						addone = true;
						done = false;
					}

					// unset this addition/deletion
					if (m_backward) {
						temp_group.set(i);
					} else {
						temp_group.clear(i);
					}
				}
			}
			if (addone) {
				if (m_backward) {
					m_best_group.clear(temp_index);
				} else {
					m_best_group.set(temp_index);
				}
				trackList.add(temp_index);
				best_merit = temp_best;
			}

		}
		m_bestMerit = best_merit;


		return Ints.toArray(trackList);
	}

	public void setSearchBackwards(boolean back) {
		m_backward = back;
	}

	/**
	 * Get whether to search backwards
	 *
	 * @return true if the search will proceed backwards
	 */
	public boolean getSearchBackwards() {
		return m_backward;
	}
	public void setNumToSelect(int n) {
		m_numToSelect = n;
	}

	/**
	 * Gets the number of attributes to be retained.
	 * @return the number of attributes to retain
	 */
	public int getNumToSelect() {
		return m_numToSelect;
	}

	/**
	 * Sets a starting set of attributes for the search. It is the
	 * search method's responsibility to report this start set (if any)
	 * in its toString() method.
	 * @param startSet a string containing a list of attributes (and or ranges),
	 * eg. 1,2,6,10-15.
	 * @throws Exception if start set can't be set.
	 */
	public void setStartSet (String startSet) throws Exception {
		m_startRange.setRanges(startSet);
	}

	/**
	 * Returns a list of attributes (and or attribute ranges) as a String
	 * @return a list of attributes (and or attribute ranges)
	 */
	public String getStartSet () {
		return m_startRange.getRanges();
	}

	/**
	 * converts the array of starting attributes to a string. This is
	 * used by getOptions to return the actual attributes specified
	 * as the starting set. This is better than using m_startRanges.getRanges()
	 * as the same start set can be specified in different ways from the
	 * command line---eg 1,2,3 == 1-3. This is to ensure that stuff that
	 * is stored in a database is comparable.
	 * @return a comma seperated list of individual attribute numbers as a String
	 */
	protected String startSetToString() {
		StringBuffer FString = new StringBuffer();
		boolean didPrint;

		if (m_starting == null) {
			return getStartSet();
		}
		for (int i = 0; i < m_starting.length; i++) {
			didPrint = false;

			if ( i != m_classIndex) {
				FString.append((m_starting[i] + 1));
				didPrint = true;
			}

			if (i == (m_starting.length - 1)) {
				FString.append("");
			}
			else {
				if (didPrint) {
					FString.append(",");
				}
			}
		}

		return FString.toString();
	}

	protected void resetOptions() {
		m_best_group = null;
		m_ASEval = null;
		m_Instances = null;
	}

}