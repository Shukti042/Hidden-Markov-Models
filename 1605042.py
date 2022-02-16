import numpy as np
import sys
import scipy.stats
import math
import random

from numpy.core.fromnumeric import argmax, shape, size


    
def main():
    stateNames=['"El Nino"','"La Nina"']
    dataFile = open(sys.argv[1], "r")
    parametersFile= open(sys.argv[2], "r")
    totalStates=int(parametersFile.readline().split()[0])
    transitionMatrix=[]
    for i in range(totalStates):
        temp=[float(x) for x in parametersFile.readline().split()]
        transitionMatrix.append(temp)
    a=np.array(transitionMatrix).T
    for i in range(a.shape[0]):
        a[i][i]=a[i][i]-1
    a[-1]=np.ones(a.shape[0])
    b=np.zeros(a.shape[0])
    b[-1]=1
    initialProbability=np.linalg.solve(a,b)
    means=[float(x) for x in parametersFile.readline().split()]
    variance=[float(x) for x in parametersFile.readline().split()]
    standardDeviations=[math.sqrt(x) for x in variance]
    parametersFile.close()
    observations=[float(x) for x in dataFile.readlines()]
    dataFile.close()
    s=np.zeros((len(observations),totalStates))
    stateTrack=np.zeros((len(observations)-1,totalStates))
    for i in range(s.shape[1]):
        # print(scipy.stats.norm(means[i],standardDeviations[i]).pdf(observations[0]))
        s[0][i]=np.log(initialProbability[i]*scipy.stats.norm(means[i],standardDeviations[i]).pdf(observations[0]))
    for observation in range(s[1:].shape[0]):
        for state in range(s.shape[1]):
            temp=[]
            for i in range(totalStates):
                # if observation==0 and state==0:
                #     print(transitionMatrix[i][state])
                #     print(scipy.stats.norm(means[state],standardDeviations[state]).pdf(observations[observation+1]))
                temp.append(s[observation][i]+np.log(transitionMatrix[i][state]*scipy.stats.norm(means[state],standardDeviations[state]).pdf(observations[observation+1])))
            s[observation+1][state]=max(temp)
            stateTrack[observation][state]=int(argmax(temp))

    # estimatedStateSequence=s.argmax(axis=1)
    outputfile=open("viterbi_output_wo_learning.txt",'w')
    # for i in estimatedStateSequence:
    #     outputfile.write(stateNames[i]+"\n")
    estimatedStateSequence=[]
    estimatedStateSequence.append(s[-1].argmax())
    for i in range(len(observations)-2,-1,-1):
        temp=int(estimatedStateSequence[-1])
        i=int(i)
        estimatedStateSequence.append(stateTrack[i][temp])
    for i in reversed(estimatedStateSequence):
        outputfile.write(stateNames[int(i)]+"\n")
    # np.savetxt("probs.txt",s)
    outputfile.close()

    threshold=0.0000000000000000
    dif=1
    random.seed(13)
    for i in range(totalStates):
        for j in range(totalStates):
            transitionMatrix[i][j]=random.uniform(0,1)
        transitionMatrix[i]=transitionMatrix[i]/np.sum(transitionMatrix[i])
        means[i]=random.uniform(0,300)
        variance[i]=random.uniform(0,20)
        standardDeviations[i]=math.sqrt(variance[i])
    # print(transitionMatrix)
    # print(means)
    # print(variance)
    while dif>threshold:
        a=np.array(transitionMatrix).T
        for i in range(a.shape[0]):
            a[i][i]=a[i][i]-1
        a[-1]=np.ones(a.shape[0])
        b=np.zeros(a.shape[0])
        b[-1]=1
        initialProbability=np.linalg.solve(a,b)
        #-------Forward--------#
        f=np.zeros((len(observations),totalStates))
        for i in range(f.shape[1]):
            f[0][i]=initialProbability[i]*scipy.stats.norm(means[i],standardDeviations[i]).pdf(observations[0])
        f[0]=f[0]/np.sum(f[0])
        for observation in range(f[1:].shape[0]):
            for state in range(f.shape[1]):
                temp=0
                for i in range(totalStates):
                    temp+=(f[observation][i]*transitionMatrix[i][state]*scipy.stats.norm(means[state],standardDeviations[state]).pdf(observations[observation+1]))
                f[observation+1][state]=temp
            f[observation+1]=f[observation+1]/np.sum(f[observation+1])
        # print(np.sum(f[-1]))
        # np.savetxt("forward.txt",f)
        #-------Backward--------#
        b=np.zeros((len(observations),totalStates))
        for i in range(b.shape[1]):
            b[-1][i]=1
        b[-1]=b[-1]/np.sum(b[-1])
        for observation in range(len(observations)-2,-1,-1):
            for state in range(b.shape[1]):
                temp=0
                for i in range(totalStates):
                    temp+=(b[observation+1][i]*transitionMatrix[state][i]*scipy.stats.norm(means[i],standardDeviations[i]).pdf(observations[observation+1]))
                b[observation][state]=temp
            b[observation]=b[observation]/np.sum(b[observation])

        # np.savetxt("backward.txt",b)
        #-------pi_star--------#
        pi_star=np.zeros((len(observations),totalStates))
        for i in range(len(observations)):
            for j in range(totalStates):
                pi_star[i][j]=f[i][j]*b[i][j]
            pi_star[i]=pi_star[i]/np.sum(pi_star[i])

        # np.savetxt("pi_star.txt",pi_star)

        #-------pi_star_star--------#
        pi_star_star=np.zeros((len(observations)-1,totalStates*totalStates))
        for i in range(len(observations)-1):
            for j in range(totalStates):
                for k in range(totalStates):
                    pi_star_star[i][j*totalStates+k]=f[i][j]*transitionMatrix[j][k]*scipy.stats.norm(means[k],standardDeviations[k]).pdf(observations[i+1])*b[i+1][k]
            pi_star_star[i]=pi_star_star[i]/np.sum(pi_star_star[i])

        # np.savetxt("pi_star_star.txt",pi_star_star)
        dif=0
        temp=np.sum(pi_star_star,axis=0)
        for i in range(totalStates):
            nsum=0
            for j in range(totalStates):
                nsum+=temp[i*totalStates+j]
            for j in range(totalStates):
                dif+=abs(transitionMatrix[i][j]-(temp[i*totalStates+j]/nsum))
                transitionMatrix[i][j]=temp[i*totalStates+j]/nsum
        temp=np.sum(pi_star,axis=0).reshape(totalStates,1)
        pisum=np.zeros((totalStates,1))
        for i in range(len(observations)):
            for j in range(totalStates):
                pisum[j][0]+=pi_star[i][j]*observations[i]
        pisum=pisum/temp
        for i in range(totalStates):
            dif+=abs(means[i]-pisum[i][0])
            means[i]=pisum[i][0]
        pisum=np.zeros((totalStates,1))
        for i in range(len(observations)):
            for j in range(totalStates):
                pisum[j][0]+=pi_star[i][j]*(observations[i]-means[j])**2
        pisum=pisum/temp
        for i in range(totalStates):
            dif+=abs(variance[i]-pisum[i][0])
            variance[i]=pisum[i][0]
        standardDeviations=[math.sqrt(x) for x in variance]
        

        
    outputfile=open("parameters_learned.txt",'w')
    outputfile.write(str(totalStates)+"\n")
    for i in range(totalStates):
        for j in range(totalStates):
            outputfile.write(str(transitionMatrix[i][j])+"  ")
        outputfile.write("\n")
    for i in range(totalStates):
        outputfile.write(str(means[i])+"  ")
    outputfile.write("\n")
    for i in range(totalStates):
        outputfile.write(str(variance[i])+"  ")
    outputfile.write("\n")
    a=np.array(transitionMatrix).T
    for i in range(a.shape[0]):
        a[i][i]=a[i][i]-1
    a[-1]=np.ones(a.shape[0])
    b=np.zeros(a.shape[0])
    b[-1]=1
    initialProbability=np.linalg.solve(a,b)
    for i in range(totalStates):
        outputfile.write(str(initialProbability[i])+"  ")
    outputfile.close()

    s=np.zeros((len(observations),totalStates))
    stateTrack=np.zeros((len(observations)-1,totalStates))
    for i in range(s.shape[1]):
        s[0][i]=np.log(initialProbability[i]*scipy.stats.norm(means[i],standardDeviations[i]).pdf(observations[0]))
    for observation in range(s[1:].shape[0]):
        for state in range(s.shape[1]):
            temp=[]
            for i in range(totalStates):
                temp.append(s[observation][i]+np.log(transitionMatrix[i][state]*scipy.stats.norm(means[state],standardDeviations[state]).pdf(observations[observation+1])))
            s[observation+1][state]=max(temp)
            stateTrack[observation][state]=int(argmax(temp))

    outputfile=open("viterbi_output_after_learning.txt",'w')
    estimatedStateSequence=[]
    estimatedStateSequence.append(s[-1].argmax())
    for i in range(len(observations)-2,-1,-1):
        temp=int(estimatedStateSequence[-1])
        i=int(i)
        estimatedStateSequence.append(stateTrack[i][temp])
    for i in reversed(estimatedStateSequence):
        outputfile.write(stateNames[int(i)]+"\n")
    outputfile.close()
if __name__ == "__main__":
    main()