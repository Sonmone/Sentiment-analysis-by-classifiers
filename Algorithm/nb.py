'''This project adopts Naive Bayes Algorithm to train a specific classifiter
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

#Txt Files
top10 = open("COMP90049-2018S1_proj2-data\\top10.txt")
most100 = open("COMP90049-2018S1_proj2-data\\most100.txt")
trainData = open("COMP90049-2018S1_proj2-data\\train_raw.txt")
devData = open("COMP90049-2018S1_proj2-data\\dev_raw.txt")
testData = open("COMP90049-2018S1_proj2-data\\test_raw.txt")

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

#train the Naive Bayes classifiter
#the parameters are training vector lists and related classes
def trainNBC(trainMatrix, trainCategory):
    numOfSample = len(trainMatrix) #the number of training data set
    numOfDic = len(trainMatrix[0]) #the length of a training data vector
    #record the number of each class's data set
    numOfClap = 0; numOfCry = 0; numOfDisappoint = 0; numOfExplode = 0; numOfFacepalm = 0
    numOfHands = 0; numOfNeutral = 0; numOfShrug = 0; numOfThink = 0; numOfUpside = 0
    #calculate the number
    for category in trainCategory:
        if category == "clap":
            numOfClap += 1
        elif category == "cry":
            numOfCry += 1
        elif category == "disappoint":
            numOfDisappoint += 1
        elif category == "explode":
            numOfExplode += 1
        elif category == "facepalm":
            numOfFacepalm += 1
        elif category == "hands":
            numOfHands += 1
        elif category == "neutral":
            numOfNeutral += 1
        elif category == "shrug":
            numOfShrug += 1
        elif category == "think":
            numOfThink += 1
        elif category == "upside":
            numOfUpside += 1
        else:
            print("unrecognized class")

    pClassList = [] #the probability of each class, such as P(clap)
    pClassList.append(numOfClap / float(numOfSample))
    pClassList.append(numOfCry / float(numOfSample))
    pClassList.append(numOfDisappoint / float(numOfSample))
    pClassList.append(numOfExplode / float(numOfSample))
    pClassList.append(numOfFacepalm / float(numOfSample))
    pClassList.append(numOfHands / float(numOfSample))
    pClassList.append(numOfNeutral / float(numOfSample))
    pClassList.append(numOfShrug / float(numOfSample))
    pClassList.append(numOfThink / float(numOfSample))
    pClassList.append(numOfUpside / float(numOfSample))

    #the numerator of the probability of P([vector] | class)
    pOfClap = ones(numOfDic); pOfCry = ones(numOfDic); pOfDisappoint = ones(numOfDic);
    pOfExplode = ones(numOfDic); pOfFacepalm = ones(numOfDic); pOfHands = ones(numOfDic);
    pOfNeutral = ones(numOfDic); pOfShrug = ones(numOfDic); pOfThink = ones(numOfDic); pOfUpside = ones(numOfDic);

    #the denominator of the probability of P([vector] | class)
    pDenomOfClap = 2.0; pDenomOfCry = 2.0; pDenomOfDisappoint = 2.0; pDenomOfExplode = 2.0; pDenomOfFacepalm = 2.0;
    pDenomOfHands = 2.0; pDenomOfNeutral = 2.0; pDenomOfShrug = 2.0; pDenomOfThink = 2.0; pDenomOfUpside = 2.0;

    #calculate the value of P([vector] | class)
    for i in range(numOfSample):
        if trainCategory[i] == "clap":
            pOfClap += trainMatrix[i]
            pDenomOfClap += sum(trainMatrix[i])
        elif trainCategory[i] == "cry":
            pOfCry += trainMatrix[i]
            pDenomOfCry += sum(trainMatrix[i])
        elif trainCategory[i] == "disappoint":
            pOfDisappoint += trainMatrix[i]
            pDenomOfDisappoint += sum(trainMatrix[i])
        elif trainCategory[i] == "explode":
            pOfExplode += trainMatrix[i]
            pDenomOfExplode += sum(trainMatrix[i])
        elif trainCategory[i] == "facepalm":
            pOfFacepalm += trainMatrix[i]
            pDenomOfFacepalm += sum(trainMatrix[i])
        elif trainCategory[i] == "hands":
            pOfHands += trainMatrix[i]
            pDenomOfHands += sum(trainMatrix[i])
        elif trainCategory[i] == "neutral":
            pOfNeutral += trainMatrix[i]
            pDenomOfNeutral += sum(trainMatrix[i])
        elif trainCategory[i] == "shrug":
            pOfShrug += trainMatrix[i]
            pDenomOfShrug += sum(trainMatrix[i])
        elif trainCategory[i] == "think":
            pOfThink += trainMatrix[i]
            pDenomOfThink += sum(trainMatrix[i])
        elif trainCategory[i] == "upside":
            pOfUpside += trainMatrix[i]
            pDenomOfUpside += sum(trainMatrix[i])

    #use log() function to transfer multiple to add
    pVecOfClap = log(pOfClap / pDenomOfClap); pVecOfCry = log(pOfCry / pDenomOfCry);
    pVecOfDisappoint = log(pOfDisappoint / pDenomOfDisappoint); pVecOfExplode = log(pOfExplode / pDenomOfExplode);
    pVecOfFacepalm = log(pOfFacepalm / pDenomOfFacepalm); pVecOfHands = log(pOfHands / pDenomOfHands);
    pVecOfNeutral = log(pOfNeutral / pDenomOfNeutral); pVecOfShrug = log(pOfShrug / pDenomOfShrug);
    pVecOfThink = log(pOfThink / pDenomOfThink); pVecOfUpside = log(pOfUpside / pDenomOfUpside);

    pVecList = [] #the final probability
    pVecList.append(pVecOfClap); pVecList.append(pVecOfCry); pVecList.append(pVecOfDisappoint);
    pVecList.append(pVecOfExplode); pVecList.append(pVecOfFacepalm); pVecList.append(pVecOfHands);
    pVecList.append(pVecOfNeutral); pVecList.append(pVecOfShrug); pVecList.append(pVecOfThink);
    pVecList.append(pVecOfUpside);

    return pVecList, pClassList

#judge the class of a test vector data
def classifyNB(targetVec, pVecList, pClassList):
    #calculate the total probability of a class P(class) * P([vector] | class)
    #use log() to transfer multiple to add
    pClap = sum(targetVec * pVecList[0]) + log(pClassList[0])
    pCry = sum(targetVec * pVecList[1]) + log(pClassList[1])
    pDisappoint = sum(targetVec * pVecList[2]) + log(pClassList[2])
    pExplode = sum(targetVec * pVecList[3]) + log(pClassList[3])
    pFacepalm = sum(targetVec * pVecList[4]) + log(pClassList[4])
    pHands = sum(targetVec * pVecList[5]) + log(pClassList[5])
    pNeutral = sum(targetVec * pVecList[6]) + log(pClassList[6])
    pShrug = sum(targetVec * pVecList[7]) + log(pClassList[7])
    pThink = sum(targetVec * pVecList[8]) + log(pClassList[8])
    pUpside = sum(targetVec * pVecList[9]) + log(pClassList[9])
    #add the probability into a dict
    dic = {'clap': pClap, 'cry': pCry, 'disappoint': pDisappoint, 'explode': pExplode, 'facepalm': pFacepalm, 'hands': pHands, 'neutral': pNeutral, 'shrug': pShrug, 'think': pThink, 'upside': pUpside}
    predict = sorted(dic, key=lambda x:dic[x])[-1] #get the predicted class with the highest possibility
    # print(pClap, pCry, pDisappoint, pExplode, pFacepalm, pHands, pNeutral, pShrug, pThink, pUpside)
    return predict

if __name__ == '__main__':
    #load the data set
    tranWordList, tranClassVec, testWordList, testClassVec, tranStringList = loadDataSet()
    #create the vocabulary dictionary
    dictionary = createDic(tranStringList)
    #get the training data set vectors
    tranVecList = produceWordFreVec(dictionary, tranWordList)
    #get the testing data set vectors
    testVecList = produceWordFreVec(dictionary, testWordList)
    #train the classifiter
    pVec, pClass = trainNBC(tranVecList, tranClassVec)

    tickCount = 0 #correct predicted number
    evaVec = zeros((10, 10)) #confusion matrix
    evaDic = {'clap': 0, 'cry': 1, 'disappoint': 2, 'explode': 3, 'facepalm': 4, 'hands': 5, 'neutral': 6, 'shrug': 7, 'think': 8, 'upside': 9}
    for i in range(len(testVecList)):
        #test a sample
        testVec = classifyNB(testVecList[i], pVec, pClass)
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
            # print("Predicting Result: Success")
        # else:
            # print("Predicting Result: Failure")
        # print("=============================================")
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
