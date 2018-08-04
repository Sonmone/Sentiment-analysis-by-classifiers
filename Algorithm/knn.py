'''This project adopts K-Nearest Neighbor Algorithm to train a specific classifiter
   which can predict a tweet with a specific emoji.
'''

#packages
from numpy import * #used for array computation
#used for TF-IDF computation
from sklearn.feature_extraction.text import TfidfTransformer
#transfor text to vectors
from sklearn.feature_extraction.text import CountVectorizer
#used to analyse vectors
from pandas import *
#import collections
import collections

#Txt Files
top10 = open("COMP90049-2018S1_proj2-data\\top10.txt")
most100 = open("COMP90049-2018S1_proj2-data\\most100.txt")
trainData = open("COMP90049-2018S1_proj2-data\\train_raw.txt")
devData = open("COMP90049-2018S1_proj2-data\\dev_raw.txt")
testData = open("COMP90049-2018S1_proj2-data\\test_raw.txt")
test = open("COMP90049-2018S1_proj2-data\\devtest.txt")

#load the data set for both training and testing
def loadDataSet():
    tranWordList = [] #the data set used for training
    tranClassVec = [] #the classes for training
    testWordList = [] #the data set used for testing
    testClassVec = [] #the classes used for training
    tranStringList = [] #the data set (String Type) used for training

    #read training data and store them in lists
    for line in trainData.readlines(): #traverse each line
        lineSplit = line.split() #get each word in a line
        lineSplit = [item.lower() for item in lineSplit] #transfor to lower case
        tranWordList.append(lineSplit[2:]) #training data
        str = ' '.join(lineSplit[2:]) #training data string
        tranStringList.append(str)
        tranClassVec.append(lineSplit[1]) #training classes

    #read testing data and store them in a list
    for line in devData.readlines(): #traverse each line
        lineSplit = line.split() #get each word in a line
        lineSplit = [item.lower() for item in lineSplit] #transfor to lower case
        testWordList.append(lineSplit[2:]) #testing data
        testClassVec.append(lineSplit[1]) #testing classes
    return tranWordList, tranClassVec, testWordList, testClassVec, tranStringList

#create the dictionary of all vocabularies
def createDic(dataSet):
    #create word frequence vectorizer
    vectorizer = CountVectorizer(stop_words='english')
    freVec = vectorizer.fit_transform(dataSet)
    #create TF-IDF transformer
    transformer = TfidfTransformer()
    tfidfVec = transformer.fit_transform(freVec)

    feature_names = vectorizer.get_feature_names() #the feature names
    freVecArray = DataFrame(freVec.toarray())
    tfidfArray = DataFrame(tfidfVec.toarray())
    #choose the 500 highest value of TF-IDF features
    tfidfFeature = tfidfArray.sum(axis=0).sort_values(ascending=False)[:500].index
    # vecRes = freVecArray.ix[:,tfidfFeature].values

    #get the dictionary
    dic = Series(feature_names)[tfidfFeature].values.tolist()
    #add top10 features
    for word in top10.readlines():
        if word not in dic:
            dic.append(word.replace("\n", ""))

    return dic

#transfer word list into vector list
def produceWordFreVec(dic, wordList):
    wordListVec = [] #sotre vectors

    for line in wordList: #traverse the word list
        lineVec = [0] * len(dic) #create the vector list with initial value of 0
        for word in line:
            if word in dic:
                lineVec[dic.index(word)] += 1 #the frequence of the word
        wordListVec.append(lineVec)
    return wordListVec

def getDistance(trainVec, targetVec):
    npvec1, npvec2 = array(trainVec), array(targetVec)
    return math.sqrt(((npvec1-npvec2)**2).sum())

def orderDic(dic, reverse):
    orderedList = sorted(
        dic.items(), key=lambda item: item[1], reverse=reverse)
    return orderedList

#judge the class of a test vector data
def classifyKNN(trainVecList, trainClassList, targetVec, k):
    globalDistance = [] #store all the distances of training vectors
    for i in range(len(trainVecList)):
        localDistance = [] #store the current distance
        distance = getDistance(trainVecList[i], targetVec)
        localDistance.append(trainClassList[i])
        localDistance.append(distance)
        globalDistance.append(localDistance)
    globalDistanceArray = array(globalDistance)
    orderDistance = (globalDistanceArray.T)[1].argsort() #sort the distances
    order = array((globalDistanceArray[orderDistance].T)[0])
    topKData = array(order[:k]) #get the first k candidates
    finalClass = orderDic(collections.Counter(topKData), True)[0][0] #choose the most nearest class
    return finalClass

if __name__ == '__main__':
    #load the data set
    tranWordList, tranClassVec, testWordList, testClassVec, tranStringList = loadDataSet()
    #create the vocabulary dictionary
    dictionary = createDic(tranStringList)
    #get the training data set vectors
    tranVecList = produceWordFreVec(dictionary, tranWordList)
    #get the testing data set vectors
    testVecList = produceWordFreVec(dictionary, testWordList)

    tick = 1
    tickCount = 0 #correct predicted number
    evaVec = zeros((10, 10)) #evaluated matrix
    evaDic = {'clap': 0, 'cry': 1, 'disappoint': 2, 'explode': 3, 'facepalm': 4, 'hands': 5, 'neutral': 6, 'shrug': 7, 'think': 8, 'upside': 9}
    for i in range(len(testVecList)):
        #test a sample
        testVec = classifyKNN(tranVecList, tranClassVec, testVecList[i], 200)
        # print("Actual Class: ", end = '')
        # print(testClassVec[i])
        # print("Predicted Class: ", end = '')
        # print(testVec)
        if testClassVec[i] == "clap":
            row = evaDic[testVec]
            evaVec[row][0] += 1
        elif testClassVec[i] == "cry":
            row = evaDic[testVec]
            evaVec[row][1] += 1
        elif testClassVec[i] == "disappoint":
            row = evaDic[testVec]
            evaVec[row][2] += 1
        elif testClassVec[i] == "explode":
            row = evaDic[testVec]
            evaVec[row][3] += 1
        elif testClassVec[i] == "facepalm":
            row = evaDic[testVec]
            evaVec[row][4] += 1
        elif testClassVec[i] == "hands":
            row = evaDic[testVec]
            evaVec[row][5] += 1
        elif testClassVec[i] == "neutral":
            row = evaDic[testVec]
            evaVec[row][6] += 1
        elif testClassVec[i] == "shrug":
            row = evaDic[testVec]
            evaVec[row][7] += 1
        elif testClassVec[i] == "think":
            row = evaDic[testVec]
            evaVec[row][8] += 1
        elif testClassVec[i] == "upside":
            row = evaDic[testVec]
            evaVec[row][9] += 1

        if testVec == testClassVec[i]:
            tickCount += 1
        print(tick)
        tick += 1
    #the accuracy of the whole predicted samples
    print("Accuracy: ", end = '')
    print("%.2f%%" % ((float(tickCount) / len(testVecList)) * 100))
    print("======================================================================")

    rowSum = evaVec.sum(axis = 1) #the sum of the rows of the evaluated matrix
    colSum = evaVec.sum(axis = 0) #the sum of the columns of the evaluated matrix
    print("Confusion Matrix:")
    print(evaVec) #print the confusion matrix
    print("======================================================================")

    #the precision of the samples of each class
    print("Precision: ")
    print("Clap: ", "%.2f%%" % ((float(evaVec[0][0]) / rowSum[0]) * 100))
    print("Cry: ", "%.2f%%" % ((float(evaVec[1][1]) / rowSum[1]) * 100))
    print("Disappoint: ", "%.2f%%" % ((float(evaVec[2][2]) / rowSum[2]) * 100))
    print("Explode: ", "%.2f%%" % ((float(evaVec[3][3]) / rowSum[3]) * 100))
    print("Facepalm: ", "%.2f%%" % ((float(evaVec[4][4]) / rowSum[4]) * 100))
    print("Hands: ", "%.2f%%" % ((float(evaVec[5][5]) / rowSum[5]) * 100))
    print("Neutral: ", "%.2f%%" % ((float(evaVec[6][6]) / rowSum[6]) * 100))
    print("Shrug: ", "%.2f%%" % ((float(evaVec[7][7]) / rowSum[7]) * 100))
    print("Think: ", "%.2f%%" % ((float(evaVec[8][8]) / rowSum[8]) * 100))
    print("Upside: ", "%.2f%%" % ((float(evaVec[9][9]) / rowSum[9]) * 100))
    print("======================================================================")

    #the recall of the samples of each class
    print("Recall: ")
    print("Clap: ", "%.2f%%" % ((float(evaVec[0][0]) / colSum[0]) * 100))
    print("Cry: ", "%.2f%%" % ((float(evaVec[1][1]) / colSum[1]) * 100))
    print("Disappoint: ", "%.2f%%" % ((float(evaVec[2][2]) / colSum[2]) * 100))
    print("Explode: ", "%.2f%%" % ((float(evaVec[3][3]) / colSum[3]) * 100))
    print("Facepalm: ", "%.2f%%" % ((float(evaVec[4][4]) / colSum[4]) * 100))
    print("Hands: ", "%.2f%%" % ((float(evaVec[5][5]) / colSum[5]) * 100))
    print("Neutral: ", "%.2f%%" % ((float(evaVec[6][6]) / colSum[6]) * 100))
    print("Shrug: ", "%.2f%%" % ((float(evaVec[7][7]) / colSum[7]) * 100))
    print("Think: ", "%.2f%%" % ((float(evaVec[8][8]) / colSum[8]) * 100))
    print("Upside: ", "%.2f%%" % ((float(evaVec[9][9]) / colSum[9]) * 100))
    print("======================================================================")

    #the f1_score of the samples of each class
    print("F1 Score: ")
    print("Clap: ", (2 * (float(evaVec[0][0]) / rowSum[0]) * (float(evaVec[0][0]) / colSum[0]) / ((float(evaVec[0][0]) / rowSum[0]) + (float(evaVec[0][0]) / colSum[0]))))
    print("Cry: ", (2 * (float(evaVec[1][1]) / rowSum[1]) * (float(evaVec[1][1]) / colSum[1]) / ((float(evaVec[1][1]) / rowSum[1]) + (float(evaVec[1][1]) / colSum[1]))))
    print("Disappoint: ", (2 * (float(evaVec[2][2]) / rowSum[2]) * (float(evaVec[2][2]) / colSum[2]) / ((float(evaVec[2][2]) / rowSum[2]) + (float(evaVec[2][2]) / colSum[2]))))
    print("Explode: ", (2 * (float(evaVec[3][3]) / rowSum[3]) * (float(evaVec[3][3]) / colSum[3]) / ((float(evaVec[3][3]) / rowSum[3]) + (float(evaVec[3][3]) / colSum[3]))))
    print("Facepalm: ", (2 * (float(evaVec[4][4]) / rowSum[4]) * (float(evaVec[4][4]) / colSum[4]) / ((float(evaVec[4][4]) / rowSum[4]) + (float(evaVec[4][4]) / colSum[4]))))
    print("Hands: ", (2 * (float(evaVec[5][5]) / rowSum[5]) * (float(evaVec[5][5]) / colSum[5]) / ((float(evaVec[5][5]) / rowSum[5]) + (float(evaVec[5][5]) / colSum[5]))))
    print("Neutral: ", (2 * (float(evaVec[6][6]) / rowSum[6]) * (float(evaVec[6][6]) / colSum[6]) / ((float(evaVec[6][6]) / rowSum[6]) + (float(evaVec[6][6]) / colSum[6]))))
    print("Shrug: ", (2 * (float(evaVec[7][7]) / rowSum[7]) * (float(evaVec[7][7]) / colSum[7]) / ((float(evaVec[7][7]) / rowSum[7]) + (float(evaVec[7][7]) / colSum[7]))))
    print("Think: ", (2 * (float(evaVec[8][8]) / rowSum[8]) * (float(evaVec[8][8]) / colSum[8]) / ((float(evaVec[8][8]) / rowSum[8]) + (float(evaVec[8][8]) / colSum[8]))))
    print("Upside: ", (2 * (float(evaVec[9][9]) / rowSum[9]) * (float(evaVec[9][9]) / colSum[9]) / ((float(evaVec[9][9]) / rowSum[9]) + (float(evaVec[9][9]) / colSum[9]))))
