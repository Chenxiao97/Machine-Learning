import java.util.*;
import java.io.*;

public class bayes{



    private static Map<String, Integer> classCounter = new HashMap<>();
    private static Map<Double, String> resultMap = new HashMap<>();

    private static List<String []> entry = new LinkedList<>();

    private static List<Tuple> attribute = new LinkedList<>();
    
    private static PrintWriter out;


    private static int correct = 0;
    private static int wrong = 0;


    public static void main(String[] args) {
        try{
            FileReader train = new FileReader(args[0]);
            FileReader testF = new FileReader(args[1]);
            out = new PrintWriter(new BufferedWriter(new FileWriter(args[2])));
            BufferedReader readerTrain = new BufferedReader(train);   
            Set<String> classLabel = new HashSet<>();
            List<String []> test = new LinkedList<>();

            String line;
            while((line = readerTrain.readLine()) != null) {
                String[] row = line.split("\\s+");
                entry.add(row);
                String s = row[0];
                int n = classCounter.getOrDefault(row[0], 0);
                classLabel.add(s);
                classCounter.put(s, n + 1);
            }

            int lim = entry.get(0).length;

            for(int i = 0; i < lim; i++){
                attribute.add(new Tuple());
            }

            for(int i = 0; i < lim; i++) {  //!!
                for(int j = 0; j < entry.size(); j++) {
                    attribute.get(i).add(entry.get(j)[i], entry.get(j)[0]);
                }
            }

            BufferedReader readerTest = new BufferedReader(testF);   
            while((line = readerTest.readLine()) != null){
                test.add(line.split("\\s+"));
            }

            nb(classLabel,test);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
        if(correct + wrong != 0){
            System.out.print("correct rate: ");

            out.print("correct rate: ");

            System.out.printf("%.2f%%\n", correct * 100.0 / (correct + wrong));

            out.printf("%.2f%%\n", correct * 100.0 / (correct + wrong));
            //out.close();
        }
        else{
            throw new ArithmeticException();
        }
    }





    public static void nb(Set<String> classLabel, List<String []> test) {
        List<String> classLabelList = new LinkedList<>(classLabel);
        for(String[] row : test) {
            double[] conditionalProb = new double[classCounter.size()];
            for(int i = 0; i < conditionalProb.length; i++){
                
                conditionalProb[i] = 1;
            
            }

            for(int i = 1; i < row.length; i++) {
                for(int j = 0; j < classLabelList.size(); j++) {
                    double p1 = (double)attribute.get(i).get(row[i], classLabelList.get(j));
                    double p2 = (double)classCounter.get(classLabelList.get(j));
                    conditionalProb[j] *=  p1 / p2;
                }
            }

            for(int i = 0; i < classLabelList.size(); i++) {
                double p2 = (double) classCounter.get(classLabelList.get(i));
                conditionalProb[i] *=  p2/ entry.size();
                resultMap.put(conditionalProb[i], classLabelList.get(i));
            }

            double max = conditionalProb[0];
            
            for(double prob : conditionalProb){
                max = Math.max(prob, max);
            }

            if(resultMap.get(max).equals(row[0])) {
                correct++;
                System.out.printf("%-10s %s\n", "Correct:", Arrays.toString(row));

                out.printf("%-10s %s\n", "Correct:", Arrays.toString(row));
            }
            else {
                wrong++;
                System.out.printf("%-10s %s\n", "Incorrect:", Arrays.toString(row));

                out.printf("%-10s %s\n", "Incorrect:", Arrays.toString(row));
            }
        }
    }


}

class Value {
    public Map<String,Integer> valueCount = new HashMap<>();  //priv?
}

class Tuple{
    private Map<String,Value> attributeMap = new HashMap<>();

    public void add(String attribute, String value) {
        if(attributeMap.containsKey(attribute)) {
            Map<String,Integer> valC = attributeMap.get(attribute).valueCount;
            attributeMap.get(attribute).valueCount.put(value, valC.getOrDefault(value,0) + 1);

        }
        else {
            attributeMap.put(attribute, new Value());
            attributeMap.get(attribute).valueCount.put(value, 1);

        }
    }

    public int get(String attribute, String value) {
        Map<String,Integer> valC = attributeMap.get(attribute).valueCount;
        int v = valC.containsKey(value) ? valC.get(value) : 1;
        return attributeMap.containsKey(attribute) ? v : 1;
    }
}