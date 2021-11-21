''''

    Author: 
        Ahmed Rafik El-Mehdi BAAHMED || BR34CH-HUNT3R


    The idea is to identify which of a set of categories (classes) points in a plan belongs to using k-NN Algorithm: 
        1. Generate two classes of point (class 1 & class 2), each class represent a circle in a plane with a specific radius.
        2. Generate the third class (class C) whose points are between the two previous classes.
        3. Classify class C points with the k-NN algorithm.
    
    PS: In k-NN classification, the output is a class membership.

'''

import random
from math import *
from matplotlib import pyplot as plt

class1 = []
class2 = []
classC = []

classCtoC1 = []
classCtoC2 = []

fig, axs = plt.subplots(2)

def generateC1C2 (R1, R2, class1, class2, classC, nbC1C2, nbC):
    '''
    ---------------------------------------------------------------------------------
        R1 -> radius of class 1 circle
        R2 -> radius of class 2 circle
    ---------------------------------------------------------------------------------
        class1 -> class 1 point list
        class2 -> class 2 point list
        classC -> class C point list
    ---------------------------------------------------------------------------------
        nbC1C2 -> number of points to generate for each class (class 1 & class 2)
        nbC -> number of points to generate for the uknown class (Class C)
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
        axs[1].plot(x1,y1, marker="o", color="red")

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
        axs[1].plot(x2,y2, marker="o", color="blue")

    for i in range(nbC):
        # -------------------------- class C ---------------------------------
        # polar notation
        # angle of the point [0, 2*pi]
        angC = random.uniform(0, 1) * 2 * pi
        # the length of the hypotenuse [0, radius]
        hypC = random.uniform(R1, R2)

        xC = cos(angC) * hypC
        yC = sin(angC) * hypC
        classC.append((xC,yC))
        axs[0].plot(xC,yC, marker="o", color="yellow")
    plt.draw()


def distance(pt1,pt2):
    return sqrt( (pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2 )


def knnClassification (k, class1, class2, classC, classCtoC1, classCtoC2):
    if (k % 2 != 0):
        figSubtitle = str(k)+'-NN Classifier'
        fig.suptitle(figSubtitle)
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
                # Trier le tableau
                classCmin[i+1].sort(key = lambda x: x[1])
                somme = 0
                for j in range(k):
                    somme = somme + classCmin[i+1][j][0]
                if (somme >= 1):
                    # class1.append((classCmin[i][0],classCmin[i][1]))
                    classCtoC1.append((classCmin[i][0],classCmin[i][1]))
                    axs[1].plot(classCmin[i][0],classCmin[i][1], marker="o", color="orange")
                else:
                    # class2.append((classCmin[i][0],classCmin[i][1]))
                    classCtoC2.append((classCmin[i][0],classCmin[i][1]))
                    axs[1].plot(classCmin[i][0],classCmin[i][1], marker="o", color="green")
        print('------------------------------------------------------------------------------------')
        print('classCtoC1 :\n', classCtoC1)
        print()
        print('classCtoC2 :\n', classCtoC2)
        print()
        print('classC points distances with the 2 other classes :\n', classCmin)
        print('------------------------------------------------------------------------------------')
    else:
        print('ERROR : k must be ODD')
    plt.draw()


if __name__ == '__main__':
    generateC1C2(10, 20, class1, class2, classC, 100, 20)
    knnClassification(3, class1, class2, classC, classCtoC1, classCtoC2)
    plt.show()
