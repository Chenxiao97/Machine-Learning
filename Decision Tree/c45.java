import java.util.*;
import java.io.*;

public class c45 {
    /**
     * General Algorithm:
     * 1. Check base cases:
     *      1.1 Base case 1: All the samples in the list belong to the same class. 
     *                       It simply creates a leaf node for the decision tree saying to choose that class. 
     *      1.2 Base case 2: None of the features provide any information gain. 
     *                       It creates a decision node higher up the tree using the expected value of the class.
     *      1.3 Base case 3: Previously-unseen class encountered. 
     *                       It creates a decision node higher up the tree using the expected value of the class.
     * 2. For each attribute a, find the normalized information gain ratio from the splitting on a.
     * 3. Let a_best be the attribute with the highest normalized information gain.
     * 4. Create a decision node that splits on a_best.
     * 5. Recur on the sublists obtained by splitting on a_best, and add those nodes as children of node.
     * 
     **/

    private static PrintWriter out;
    private static int correct = 0;
    private static int wrong = 0;
    private static TreeNode root = new TreeNode("null");


    private static List<Attribute> attributeList = new ArrayList<>();
    private static Map<Attribute, Boolean> attributeUsage = new HashMap<>();


    public static void main(String[] args) {
        if(args.length < 3){
            System.out.println("Usage: [training data file] [testing data file] [output file]");
        }
        try{
            out = new PrintWriter(new BufferedWriter(new FileWriter(args[2])));
            root.attribute = null;
            FileReader train = new FileReader(args[0]);
            preprocess(train);
            buildTree(root);


            LinkedList<TreeNode> l = new LinkedList<>();
            l.offer(root);
            while(!l.isEmpty()) {
                TreeNode node = l.poll();
                if(node.attribute != null)
                    out.printf("attribute: %d ,value: %s\n", node.attribute.attrNumber, node.val);

                for(int i = 0; i < node.children.size(); i++) {
                    l.offer(node.children.get(i));
                }
            }

            FileReader testF = new FileReader(args[1]);
            eval(testF);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        if(correct + wrong != 0){
            out.printf("%.2f%%\n", correct * 100.0 / (correct + wrong));
            out.close();
        }
    }

    public static void preprocess(FileReader trainingFile) throws IOException{
        BufferedReader br = new BufferedReader(trainingFile);
        String line;
        List<String[]> iniEntry = new ArrayList<>();
        while((line = br.readLine()) != null) {
            String[] row = line.split("\\s+");
            iniEntry.add(row);

            if(iniEntry.size() == 1) {
                for (int i = 0; i < row.length - 1; i++) {
                    Attribute a = new Attribute(i+1);
                    attributeList.add(a);
                    attributeUsage.put(attributeList.get(attributeList.size() - 1), true);
                }
            }
        }

        for(String[] row : iniEntry) {
            String s = row[0];
            for(int i = 0; i < row.length - 1; i++) {
                attributeList.get(i).add(row[i + 1], s);
            }

        }
        root.entry = iniEntry;

    }





    public static Attribute nextAttr(Map<String, Integer> classCount, List<String[]> partition) {

        double sum1 = 0.0;
        for(Integer i : classCount.values()){
            sum1 += (double)i;
        }
        double classInfo = 0.0;
        for(Integer i : classCount.values()){
            Double num = (double)i;
            classInfo += (-1.0 * num / sum1 ) * (Math.log(num / sum1) / Math.log(2));
        }

        double maxGainRatio = (double)Integer.MIN_VALUE;
        Attribute nextAttr = null;

        for(Attribute a : attributeList) {
            double attributeInfo = 0;
            double splitInfo = 0;
            if(attributeUsage.get(a) == false){
                continue;
            }
            for(String s : a.valueMap.keySet()) {
                List<Integer> count = new LinkedList<>(a.valueMap.get(s).count.values());
                int sum = 0;
                for(int n : count){
                    sum += n;
                }
                double part1 = Math.log(1.0 * sum / partition.size());
                double part2 = Math.log(2);
                splitInfo += ((double)-1 * sum / partition.size()) * ( part1 / part2);
                double sum2 = 0.0;
                for(Integer i : count){
                    sum2 += (double)i;
                }
                double info2 = 0.0;
                for(Integer i : count){
                    Double num = (double)i;
                    info2 += (-1.0 * num / sum2 ) * (Math.log(num / sum2) / Math.log(2));
                }
                attributeInfo += ((double)sum / partition.size()) * info2;
             }

             double gainRatio = (classInfo - attributeInfo) / (splitInfo + 1 / Double.MAX_VALUE);
             
             if(gainRatio <= maxGainRatio){
                continue;
             }
             else{
                 maxGainRatio = gainRatio;
                 nextAttr = a;
             }
        }
        return nextAttr;
    }


    public static void buildTree(TreeNode node) {

        if(node.entry.size() == 0){
            return;
        }
        boolean avail = false;
        for(Attribute a : attributeUsage.keySet()) {
            if(attributeUsage.get(a) == true){
                avail = true;
                continue;
            }
        }
        if(avail == false){
            return;
        }
        if(node.val.equals("null") == false) {
            for(Attribute a : attributeList) {
                if(attributeUsage.get(a) == true){
                    a.valueMap.clear();
                }
            }
        }
        Map<String, Integer> classCount = new HashMap<>();

        for(String[] row : node.entry) {
            if(node.val.equals("null") == false) {
                for(int i = 0; i < row.length - 1; i++) {
                    Attribute a = attributeList.get(i);
                    String s1 = row[i+1];
                    if(attributeUsage.get(a) == true){
                        a.add(s1, row[0]);
                    }
                }
            }
            classCount.put(row[0], classCount.getOrDefault(row[0], 0) + 1);
        }
        if(classCount.size() == 1) {
            ArrayList<String> temp = new ArrayList<>(classCount.keySet());
            node.classification = temp.get(0); 
            return;
        }

        Attribute nextAttr = nextAttr(classCount, node.entry);
       
        if(nextAttr == null){
            return;
        }else{
            attributeUsage.put(nextAttr, false);
        }
        for(String s : nextAttr.valueMap.keySet()) {
            TreeNode childNode = new TreeNode(s);
            childNode.attribute = nextAttr;
            node.children.add(childNode);
        }

        for(String[] row : node.entry) {
            List<TreeNode> children1 = node.children;
            for(TreeNode childNode : children1) {
                String atr = row[nextAttr.attrNumber];
                if(atr.equals(childNode.val) == true){
                    childNode.entry.add(row);
                }
            }
        }
        for(TreeNode childNode : node.children){
            buildTree(childNode);
        }
    }



    public static void eval(FileReader testF) throws IOException{

        BufferedReader reader = new BufferedReader(testF);
        List<String[]> testList = new ArrayList<>();

        String line;
        while((line = reader.readLine()) != null) {
            testList.add(line.split("\\s+"));
        }

        for(String[] str : testList){
            evalHelper(root,str);
        }
    }

    
    public static void evalHelper(TreeNode root, String[] row) {
        if(root.classification != null) {
            if(root.classification.equals(row[0])) {
                correct++;
                out.printf("Correct: %s\n", Arrays.toString(row));
            }
            else {
                wrong++;
                out.printf("Wrong: %s\n", Arrays.toString(row));
            }
            return;
        }

        for(TreeNode childNode : root.children){
            if(row[childNode.attribute.attrNumber].equals(childNode.val)){
                evalHelper(childNode, row);
            }
        }
    
    }

}


class Value{
    Map<String, Integer> count = new HashMap<>();
}

class Attribute{
    public int attrNumber;
    Map<String, Value> valueMap;


    public Attribute(int attrNumber){
        this.valueMap = new HashMap<>();
        this.attrNumber = attrNumber;
    }

    //key is the value under attribute and cat is the classification of the entry
    public void add(String key, String cat) {  
        if(valueMap.containsKey(key) == false) {
            valueMap.put(key, new Value());
        }

        Value v = valueMap.get(key);
        v.count.put(cat, v.count.getOrDefault(cat, 0) + 1);
    }
}


class TreeNode{

    public String val;
    public String classification;
    public Attribute attribute;
    public TreeNode(String val) {
        this.val = val;
    }
    public List<TreeNode> children = new LinkedList<>();
    public List<String[]> entry = new LinkedList<>();

}