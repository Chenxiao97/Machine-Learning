/*author: Chenxiao Wang*/
/*THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - CHENXIAO WANG*/
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class Apriori{
	private static int MIN_SUPPORT;
	private static PrintWriter out;


	private static class Candidate {
		int support;
		int[] set;

		Candidate(int support, int[] set) {
			this.support = support;
			this.set = set;
		}
	}

	private static List<Map<String, Candidate>> list = new ArrayList<>();


	public static void main(String[] args) {
		/*The program should be executable with 3 parameters: 
		 * the name of the input dataset file, 
		 * the threshold of minimum support count, 
		 * and the name of the output file (in that order). 
		 * The minimum support count should be an integer. 
		 * An itemset is frequent if its support count is larger or equal to this threshold. */

		if(args.length < 3){
			System.out.println("Insufficient Inputs");
		}

		MIN_SUPPORT = Integer.parseInt(args[1]);
		String outputFile = args[2];
		
		try{
			PrintStream terminal = System.out;
			PrintStream redirect = new PrintStream(new FileOutputStream(outputFile));
			System.setOut(redirect);
			long timer = new Date().getTime();

			FileReader in = new FileReader(args[0]);
			out = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)));

			AprioriPruning(in);
			System.setOut(terminal);

			System.out.printf("%f seconds", (new Date().getTime()-timer) / 1000.0);

			FileReader out_2 = new FileReader(args[2]);
			sortOut(out_2, outputFile, terminal);
		}
		catch(Exception e) {
			System.out.println("Error in finding files");
		}

	}



	public static void AprioriPruning(FileReader inputFile) throws Exception{
		BufferedReader reader = new BufferedReader(inputFile);
		Map<String, Candidate> singleSet = new ConcurrentHashMap<>();
		Map<String, Candidate> twoSet = new ConcurrentHashMap<>();
		List<List<Integer>> buffer = new ArrayList<>();
		String line;
		while((line = reader.readLine()) != null) {
			List<Integer> temp = new ArrayList<Integer>();
			String[] str = line.trim().split("\\s+");
			for(int i = 0; i < str.length; i++){
					temp.add(Integer.parseInt(str[i]));
			}
	
			//now we can generate the single item set
			for(int i = 0; i < temp.size(); i++){
				int[] set = new int[]{temp.get(i)};
				String key = Arrays.toString(set);
				if(!singleSet.containsKey(key)){
					singleSet.put(key, new Candidate(1, set));
				}
				else{
					singleSet.get(key).support++;			
				}	

			}


			//get all 2-item set
			for(int i = 0; i < temp.size() - 1; i++) {
				for(int j = i+1; j < temp.size(); j++) {
					int[] set = new int[]{temp.get(i), temp.get(j)};
					String key = Arrays.toString(set);
					if(!twoSet.containsKey(key))
						twoSet.put(key, new Candidate(1, set));
					else
						twoSet.get(key).support++;
				}
			}
			buffer.add(temp);

		}

		Map<String, Candidate> clSingle = clean(singleSet);
		list.add(clSingle);
		Map<String, Candidate> clTwo = clean(twoSet);

		list.add(clTwo);

		//Map<String, Candidate> nextSet = generateCand(list.get(list.size() - 1));
		Map<String, Candidate> nextSet = generateCand(clTwo);

		list.add(nextSet);
		while(nextSet.size() > 0) {
			testMinSupport(buffer, nextSet);
			clean(nextSet);
			Map<String, Candidate> ttmp = list.get(list.size() - 1);
			nextSet = generateCand(ttmp);
			list.add(nextSet);
		}
	}


	public static void sortOut(FileReader inFile, String inputFile, PrintStream terminal) throws Exception {
		BufferedReader reader = new BufferedReader(inFile);
		List<String> outputBuffer = new ArrayList<String>();
		
		String line;
		while((line = reader.readLine()) != null){
			outputBuffer.add(line);
		}

		Collections.sort(outputBuffer, new Comparator<String>(){
			@Override
			public int compare(String s1, String s2) {

				int[] list1 = parseLine(s1);
				int[] list2 = parseLine(s2);
				int m = 0, n = 0;
				while(m < list1.length || n < list2.length) {

					int v1, v2;
					if(m < list1.length){
						v1 = list1[m];
					}else{
						v1 = 0;
					}

					if(n < list2.length){
						v2= list2[n];
					}else{
						v2 = 0;
					}

					if(v1 != v2){
						return v1 - v2;
					}
					m++; n++;
				}
				return 0;
			}
		});

		PrintStream redirect = new PrintStream(new FileOutputStream(inputFile));
		System.setOut(redirect);
		for(String ss : outputBuffer){
			System.out.println(ss);
		}
		System.setOut(terminal);
	}



	public static int[] parseLine(String line) {
		String[] str = line.trim().substring(0, line.lastIndexOf(" ")).split("\\s+");
		int[] list = new int[str.length];
		for(int i = 0; i < list.length; i++){
			list[i] = Integer.parseInt(str[i]);
		}
				//System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

		return list;
	}

	public static Map<String, Candidate> clean(Map<String, Candidate> map) {
		for(Map.Entry<String, Candidate> entry : map.entrySet() ) {
			if(entry.getValue().support >= MIN_SUPPORT){
				for(int i : entry.getValue().set){
					System.out.printf("%d ",i);
				}
				//System.out.printf(Arrays.toString(entry.getValue().set));
				System.out.printf("(%d)\n",entry.getValue().support);
			}
			else{
				map.remove(entry.getKey());
			}

		}
		return map;
	}


	public static void testMinSupport(List<List<Integer>> buffer, Map<String, Candidate> nextSet) {
		for(List<Integer> entry : buffer) {
			Set<Integer> set = new HashSet<Integer>(entry);
			boolean contain;

			for(Map.Entry<String, Candidate> m : nextSet.entrySet()) {
				contain = true;
				for(int i : m.getValue().set) {
					if(!set.contains(i)){
						contain = false;
					}
				}
				if(contain == true){
					m.getValue().support++;
				}
			}
		}
	}





	public static Map<String,Candidate> generateCand(Map<String, Candidate> lastSet) {
		ConcurrentHashMap<String, Candidate> nextSet = new ConcurrentHashMap<String, Candidate>();

		boolean bool;
		for(Candidate c : lastSet.values()) {
			for(Candidate d : lastSet.values()) {
				if(c == d){
					continue;
				}
				bool = true;

				for(int i = 0; i < d.set.length - 1; i++) {
					if(c.set[i] != d.set[i]){
						bool = false;
						break;
					}
				}
				if(!bool){
					continue;
				}
				
					
				int[] nextCand = new int[c.set.length + 1];
				
				for(int i = 0; i < c.set.length; i ++){
					nextCand[i] = c.set[i];
				}
				nextCand[c.set.length] = d.set[d.set.length - 1];

				Arrays.sort(nextCand);

				List<List<Integer>> subset = new ArrayList<List<Integer>>();
				recurSub(subset, new ArrayList<>(), 0, nextCand, nextCand.length - 1);
				boolean isNext = true;
				for (int i = 0; i < subset.size(); i++) {
					String ss = Arrays.toString(subset.get(i).stream()
															.mapToInt(x -> x).toArray());					
					if (lastSet.containsKey(ss)){
						isNext = true;
					}else{
						isNext = false;
					}
				}
				if (isNext)
					nextSet.put(Arrays.toString(nextCand), new Candidate(0, nextCand));
				
			}
		}
		return nextSet;
	}

	public static void recurSub(List<List<Integer>> list, ArrayList<Integer> temp, int start, int[] item, int length) {
		if(temp.size() == length){
			list.add(new ArrayList<Integer>(temp));
			return;
		}
		for(int i = start; i < item.length; i++) {
			temp.add(item[i]);
			recurSub(list,temp,i+1,item,length);
			temp.remove(temp.size() - 1);
		}
		
	}


}

