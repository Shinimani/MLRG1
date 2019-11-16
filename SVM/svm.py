import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

trainfile = sys.argv[1]
testfile = sys.argv[2]
testpredfile = sys.argv[3]
np.random.seed(0)
trainD = np.array(pd.read_csv(trainfile, header=None, delimiter=','))
print(trainD.shape)


# this class is of binary classification
class SVMbin():
    def __init__(self, x=np.empty((1, 1)), iterations=100, lambda1=1):
        # X = x
        self.iterations = iterations
        self.lambda1 = lambda1
        # here, Xb contains the 784 columns of X, and the final column is 1, for the bias term
        self.Xb = np.ones((x.shape[0], x.shape[1]))
        self.Xb[:, :-1] = x[:, :-1]
        # uniq contains the 2 unique values of X[:,-1]
        self.uniq = np.unique(x[:, -1])

        def f(z):
            if z == self.uniq[0]:
                return 1
            else:
                return -1

        self.Y = np.array([f(xi) for xi in x[:, -1]])

        # print(self.Y[-50:-1])
        self.w = np.zeros((1, self.Xb.shape[1]))

    # assuming the final column contains the labels
    # this function will train w of dimension 1,785
    def train(self, k):

        for i in range(1, self.iterations + 1):
            indices = np.random.random_integers(self.Xb.shape[0] - 1, size=(k, 1))

            # tempPlusX = np.empty(shape=(1,self.Xb.shape[1]),dtype=int)
            # tempPlusY = np.empty(shape=(1,1),dtype=int)
            #
            sum = np.zeros((1, self.Xb.shape[1]))
            for j in indices:
                y = self.Y[j]
                x = self.Xb[j, :]
                if (y * (np.dot(x, self.w.T))) < 1:
                    sum += y * x
                    # tempPlusX = np.vstack((tempPlusX,x))
                    # tempPlusY = np.vstack((tempPlusY,y))

            eta = 1 / (self.lambda1 * i)
            self.w = (1 - eta * self.lambda1) * self.w + (eta / k) * sum



        del self.Xb

        # return w

    #     assuming X is of dimensions (:,785)
    def predict(self, X):
        temp = np.ones((X.shape[0], X.shape[1]))
        temp[:, :-1] = X[:, :-1]

        def helper(t):
            td = np.dot(t, self.w.T)
            if td > 0:
                return self.uniq[0]
            else:
                return self.uniq[1]

        # helpervec = np.vectorize(helper())

        # y = helpervec(temp)
        # return y
        return np.array([helper(xi) for xi in temp])


def accuracyCalc(actual, pred):
    sum = 0
    for i in range(actual.shape[0]):
        if actual[i] == pred[i]:
            sum += 1
    return (sum / actual.shape[0])


# class for multi class SVM
class SVMmulti():
    def __init__(self, trainD, iterations=100, lambda1=1):
        # self.data = data
        self.iterations = iterations
        self.lambda1 = lambda1
        # the list of unique classes of the data
        self.uniq = np.sort(np.unique(trainD[:, -1]))
        # number fo samples in data
        self.samples = trainD.shape[0]
        # number of classes in the data
        self.classes = self.uniq.size
        # the list of individual binary models
        self.models = [[SVMbin() for j in range(i + 1, 10)] for i in range(10)]
        # for i in range(self.classes):
        #     for j in range(i+1,self.classes):
        #         tempD = np.vstack((self.data[i],self.data[j]))
        #         temp = SVMbin(x=tempD,iterations=100,lambda1=1)
        #         temp.train(k=k)
        #         self.models[i][j] = temp
        # assorted data list
        self.trainD = trainD

        #
        # self.data[0] = trainD[trainD[:,-1] == 0]
        # self.data[1] = trainD[trainD[:,-1] == 1]
        # self.data[2] = trainD[trainD[:,-1] == 2]
        # self.data[3] = trainD[trainD[:,-1] == 3]
        # self.data[4] = trainD[trainD[:,-1] == 4]
        # self.data[5] = trainD[trainD[:,-1] == 5]
        # self.data[6] = trainD[trainD[:,-1] == 6]
        # self.data[7] = trainD[trainD[:,-1] == 7]
        # self.data[8] = trainD[trainD[:,-1] == 8]
        # self.data[9] = trainD[trainD[:,-1] == 9]
        #

    def train(self, k):
        for i in range(self.classes):
            for j in range(i + 1, self.classes):
                tempD = np.vstack((self.trainD[trainD[:, -1] == i], self.trainD[trainD[:, -1] == j]))
                print(i, " before wala ij ", j)
                temp = SVMbin(x=tempD, iterations=self.iterations, lambda1=self.lambda1)
                print(i, " after wala ij ", j)

                temp.train(k=k)
                self.models[i][j - i - 1] = temp
        del self.trainD

    def predict(self, X):
        predictions = np.empty(shape=(X.shape[0], 45), dtype=int)
        print(predictions.shape, "predictions ka shape")
        k = 0
        for i in range(self.classes):
            for j in range(i + 1, self.classes):
                temp = self.models[i][j - i - 1]
                preds = temp.predict(X)
                predictions[:, k] = preds
                k += 1
        finalPreds = np.empty(shape=(X.shape[0], 1), dtype=int)
        for i in range(predictions.shape[0]):
            counts = np.bincount(predictions[i])
            finalPreds[i] = np.argmax(counts)

        return finalPreds


# a1 = np.ones(shape=(5,2))
# a2 = np.ones(shape=(5,2))
# a3 = np.append(a1,a2,axis=1)
# print(a3.shape)


model = SVMmulti(trainD=trainD, iterations=500)
model.train(k=70)
# trpreds = trial.predict(X=trainD)
# print(accuracyCalc(trainD[:,-1],trpreds))


testD = np.array(pd.read_csv(testfile, header=None, delimiter=','))
print(testD.shape)

# labels = pd.read_csv('test_labels.txt', sep="\n", header=None)
# print(labels.shape)
testpred = model.predict(X=testD)
print(testpred.shape)

# testLabels = np.array(labels)
# print(testLabels.shape)

# print(accuracyCalc(testLabels,testpred))

np.savetxt(testpredfile, testpred)
