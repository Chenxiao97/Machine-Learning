#Author: Chenxiao Wang
import csv
import numpy as np 
import sys,copy,random


def k_means(k, x):
    partitions = [] 
    indices = set()
    lens = [i for i in range(len(x)-1)]
    for i in range(k):
        index = random.choice(lens)
        while index in indices:
            index = random.choice(lens)
        partitions.append(x[index])
        indices.add(index)
    partitions = np.array(partitions)
    partitions_prev = np.zeros(partitions.shape)
    clusters = np.zeros(len(x))

    diff = 0 
    for i in range(len(partitions)):
        diff += np.linalg.norm(partitions[i] - partitions_prev[i],2)

    while diff != 0:
        for i in range(len(x)):
            list = []
            for j in range(len(partitions)): 
                list.append(np.linalg.norm(x[i] - partitions[j],2))
            index = np.argmin(list)
            clusters[i] = index

        partitions_prev = copy.deepcopy(partitions)

        for i in range(k):
            partitions[i] = np.mean(x[clusters == i], axis = 0)

        diff = 0
        for i in range(len(partitions)):
            diff += np.linalg.norm(partitions[i] - partitions_prev[i],2)

    return clusters, partitions

def sumSquaredError(k, x, clusters, partitions):
    sse = 0.0
    for i in range(k):
        for xi in x[clusters == i]:
            sse += np.linalg.norm(xi-partitions[i],2) ** 2
    return sse

def main():
    if len(sys.argv) != 4:
        print("Usage: [input data file], [k value], [output data file]")
    inFile = sys.argv[1]
    k = sys.argv[2]
    k = int(k) 
    outFile = sys.argv[3]

    reader = csv.reader(open(inFile))
    df = np.array([row for row in reader if row])
    x, y = df[:,:-1],df[:,-1]
    x = x.astype(np.float)

    clusters, centroids = k_means(k, x)

    sse = 0
    sse = sumSquaredError(k,x,clusters,centroids)
    print("SSE: %.2f" %sse)

    clusters = list(map(int, clusters))

    with open(outFile,'w') as f:
        for i in clusters:
            f.write(str(i)+'\n')
        f.write("SSE: %.3f" % sse)       

    minK = 5
    maxK = 100
    interval = 5
    k_list = (maxK - minK)/interval

if __name__ == '__main__':
    main()
