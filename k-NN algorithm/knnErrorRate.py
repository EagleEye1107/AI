''''

    Author: 
        Ahmed Rafik El-Mehdi BAAHMED || BR34CH-HUNT3R


    The idea is to calculate the Mean Error Rates to find the optimal value of k. Based on the previous principle
    in knnClassifier using two classes
    
    PS: In k-NN classification, the output is a class membership.

'''

import random
from math import *
import numpy as np
from matplotlib import pyplot as plt

class1 = []
class2 = []
classTest = []
classC = []

classCtoC1 = []
classCtoC2 = []

fig, axs = plt.subplots(2)
figSubtitle = 'k-NN Error Rate'
fig.suptitle(figSubtitle)


def generateC1C2 (R1, R2, class1, class2, classTest, classC, nbC1C2):
    '''
    ---------------------------------------------------------------------------------
        R1 -> radius of class 1 circle
        R2 -> radius of class 2 circle
    ---------------------------------------------------------------------------------
        class1 -> class 1 point list
        class2 -> class 2 point list
    ---------------------------------------------------------------------------------
        nbC1C2 -> number of points to generate for each class (class 1 & class 2)
    ---------------------------------------------------------------------------------
    '''
    for i in range (nbC1C2):
        # -------------------------- class 1 ---------------------------------
        # polar notation
        # angle of the point [0, 2*pi]
        ang1 = random.uniform(0, 1) * 2 * pi
        # the length of the hypotenuse [0, radius]
        hyp1 = sqrt(random.uniform(0, 1)) * R1

        x1 = cos(ang1) * hyp1
        y1 = sin(ang1) * hyp1
        class1.append((x1,y1))
        axs[0].plot(x1,y1, marker="o", color="red")

        # -------------------------- class 2 ---------------------------------
        # polar notation
        # angle of the point [0, 2*pi]
        ang2 = random.uniform(0, 1) * 2 * pi
        # the length of the hypotenuse [0, radius]
        hyp2 = random.uniform(R2, 1.5*R2)

        x2 = cos(ang2) * hyp2
        y2 = sin(ang2) * hyp2
        class2.append((x2,y2))
        axs[0].plot(x2,y2, marker="o", color="blue")
    
    # Test Set 20%
    for i in range((int(nbC1C2 * 0.2))):
        pt = int(random.uniform(0, nbC1C2))
        classTest.append(('C1', class1[pt]))
        classC.append(class1[pt])
        axs[0].plot(class1[pt][0],class1[pt][1], marker="o", color="orange")

        classTest.append(('C2', class2[pt]))
        classC.append(class2[pt])
        axs[0].plot(class2[pt][0],class2[pt][1], marker="o", color="green")

    classTest.sort(key = lambda x: x[0])
    plt.draw()


def distance(pt1,pt2):
    return sqrt( (pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2 )

def knnClassification (k, class1, class2, classC, classCtoC1, classCtoC2):
    if (k % 2 != 0):
        classCtoC1.clear()
        classCtoC2.clear()
        classCmin = []
        subClassCmin = []
        for pt in classC:
            classCmin.append(pt)
            for j in class1:
                subClassCmin.append((1, distance(pt,j)))
            for j in class2:
                subClassCmin.append((-1, distance(pt,j)))
            classCmin.append(subClassCmin)
            subClassCmin = []
        
        # pair (0,2..) -> point  ||  impair (1,3..) -> ses distances [Trier le tableau]
        for i in range(0, len(classCmin)):
            if (i % 2 == 0):
                classCmin[i+1].sort(key = lambda x: x[1])
                somme = 0
                for j in range(k):
                    somme = somme + classCmin[i+1][j][0]
                if (somme >= 1):
                    classCtoC1.append(('C1', (classCmin[i][0],classCmin[i][1])))
                else:
                    classCtoC2.append(('C2', (classCmin[i][0],classCmin[i][1])))
    else:
        print('ERROR : k must be ODD')


def knnOptimalK (n, class1, class2, classTest, classC, classCtoC1, classCtoC2):
    # maeList of all Mean Error calculated for each k value in range(n+1)
    maeList = []
    # graph coordinates
    x = []
    y = []

    for k in range(3, n+1):
        mae = 0
        classAfterTest = []
        if (k % 2 != 0):
            knnClassification (k, class1, class2, classC, classCtoC1, classCtoC2)
            for i in range(len(classCtoC1)):
                classAfterTest.append(classCtoC1[i])
            for i in range(len(classCtoC2)):
                classAfterTest.append(classCtoC2[i])

            # calculate MAE of the current k value
            for pt in classAfterTest:
                if pt not in classTest:
                    mae = mae + 1
            
            # insert each MEA with its k in the meaList
            maeList.append((k,mae))
    
    # create x-axis and y-axis of the graph
    for i in range(len(maeList)):
        x.append(maeList[i][0])
        y.append(maeList[i][1])

    plt.plot(x, y, 'o:b')
    plt.xlabel('K - axis')
    plt.xticks(np.arange(np.array(x).min(), np.array(x).max()+1, 2))
    plt.ylabel('MAE - axis')
    plt.yticks(np.arange(np.array(y).min(), np.array(y).max()+1, 1))
    plt.draw()

    # sort the Mean Error list to get the best value of k (for the min MAE)
    maeList.sort(key = lambda x: x[1])
    return maeList[0][0]


if __name__ == '__main__':
    generateC1C2(15, 17, class1, class2, classTest, classC, 100)
    k = knnOptimalK(21, class1, class2, classTest, classC, classCtoC1, classCtoC2)
    print("the ideal k is : ", k)
    plt.show()


'''
        Graph:
            red points : class1
            blue points : class2
            orange points : Test set of classe1
            green points : Test set of classe2
'''