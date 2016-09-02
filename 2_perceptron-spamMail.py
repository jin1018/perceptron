"""
Implementation of the Perceptron Algorithm & Filtering Spam email
Binary Classification

data files: spam_train.txt, spam_test.txt (from the SpamAssassin Public Corpus)
emails are preprocessed: lower-casting, remove html tags, normalize urls, words reduced to stem form
1 = spam, 0 = Not spam (Later on, -1 = Not spam)

The premise is that certain words (for example, "free") tend to appear frequently in spam emails
"""

#import matplotlib #use for plotting
from __future__ import division
import re #regrex
import numpy as np #matrix multiplication
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]
import random
import copy

#Split training data (spam train.txt) in to traing set(4000) and validate set(1000)
trainSet=[]
valSet=[]
with open("data/spam_train.txt") as f:
	for i, line in enumerate(f):
		if i<4000:
			trainSet.append(line)
		else:
			valSet.append(line)
f.close()
#print(trainSet[1])


#Build a vocabulary list(all words that appear) ONLY with the training set
#(validation set and test set should be completely unseen)
#ignore words that appear less than 30 times (to prevent overfitting)
#dictionary = {key:value, key:value, key:value, ...}
dict_train={}#{word:frequency}
for e in trainSet:#for each email
	e=re.sub(r'\d+', '', e)#remove 1,0
	words_in_email = e.split()#a list where each element is a word in an email
	for w in words_in_email:
		if dict_train.has_key(w):
			freq = dict_train.get(w)
			freq = freq+1
			dict_train.update({w:freq})
		else:
			dict_train[w]=1

	#delete words that appear less than 30 times (or any certain number of times)
	wordsToBeDeleted=[]
	for w in dict_train:
		if dict_train.get(w)<30:
			wordsToBeDeleted.append(w)
	for w in wordsToBeDeleted:
		del(dict_train[w])

def intoFeatureVectors(data):
	"""
	transform emails into a feature vectors
	lenght of vector = length of dictionary
	for each email, each element of the vector is wheter or not a word is in the email
	"""
	fv=[]#list of all the featur vectors of the data set 
	for e in data: #for each email
		vectorForEachEmail=[]#the length equals the lenth of the vocabulary list
		#first character of e is the y(either 1 or 0 depending on whether its spam or ham)
		#first column of the feature vector is either 0 or 1
		vectorForEachEmail.append(int(e[0]))
		for w in dict_train:
			w= " " + w + " "
			if w in e:
				vectorForEachEmail.append(1)
			else:
				vectorForEachEmail.append(0)
		fv.append(vectorForEachEmail)
		#length of fv is 434 for the data set I am using
	return fv

trainFeatureVectors = intoFeatureVectors(trainSet)#list of all the 4000 features vectors of the training set
valFeatureVectors = intoFeatureVectors(valSet)


#Implement my own Perceptron Algorithm
def perceptron_train(data):#data = trainFeatureVectors
	"""
	input data: list of all the feature vectors (4000) & y values (the correct values)

	return w: the final classification vector
	return k: number of updates (mistakes)
	return iter_passes: number of passes through the data

	Assumes that input data is linearly separable.
	So algorithm stops when all points are correctly classified

	the Perceptron is an Online algorithm : use one data example at a time, Not the entire data set
	"""

	#w: initialize the weight vector with all 0s
	w= [0]*(len(data[0])-1)#delete 1 because the first column is y value
	"""
	#OR initialize w with random numbers
	for i in w:
		w[i]=(random.randint(-10000, 10000))
	"""
	#k: number of updates
	k=0
	#number of passes through the data
	iter_passes=0

	#y: whether the email is spam or not (first column of data)
	y=[]
	for i in data:#for each feature vector
		if i[0]==1:
			y.append(1)
		elif i[0]==0:
			y.append(-1)
		del(i[0])
	#print(y)


	#Forward Propagation: use matrices to go through multiple vectors(inputs) at once, not one at a time, to speed up the process
	#DID NOT use forward propagation here

	print("start training")
	correctlyClassified=0
	while(correctlyClassified==0):
		print("iterations (iter_passes):" + str(iter_passes))

		num_correct=0
		for i in xrange(0,len(data)):
			#predicted y value(yHAT)
			if np.dot(w,data[i]) < 0:
				yHAT = -1
			else:
				yHAT = 1
			"""
			(Notes)
			Similarly in Logistic Regression,
			yHat =1 when w*x>0
			yHat =0 when w*x<0
			because of the Sigmoid function
			"""

			#check if current w correctly classifies data
			if yHAT == y[i]:#correctly classified
				num_correct= num_correct+1
			else :#Not correctly classified
				w = w + np.dot(data[i],y[i])#update weights
				"""
				(Notes)
				consider vectors visually
				dot product(projection) : A dot B = A * B * cos(angle)
				when 0<angle<90, cos(angle)>0
				when 90<angle<180, cos(angle)<0
				if yHAT is 1 but the actual y value is -1, change weights to a bigger angle over 90(and smaller magnitude)
				if yHAT is -1 but the actual y value is 1, change weights to a smaller angle smaller than 90(and greater magnitude)
				"""

				k=k+1#number of updates (mistakes)

		print("number of correctly classified: " + str(num_correct))

		if num_correct>(len(data)-10):#check if all data is correctly classified
			#-10 because the data I'm working with does not seem to converge
			correctlyClassified=1

		iter_passes=iter_passes+1


	returnArray=[]
	returnArray.append(w)
	returnArray.append(k)
	returnArray.append(iter_passes)

	return returnArray

	print("finished training")


def perceptron_validation(valSet, trainedWeights):
	"""return validation error(ValError)"""
	#y: whether the email is spam or not (first column of data)
	y=[]
	for i in valSet:#for each feature vector
		if i[0]==1:
			y.append(1)
		elif i[0]==0:
			y.append(-1)
		del(i[0])

	number_correclyClassified=0

	for i in xrange(0,len(valSet)):
		if np.dot(trainedWeights,valSet[i]) < 0:
			yHAT = -1
		else:
			yHAT = 1

		if yHAT == y[i]:
			number_correclyClassified = number_correclyClassified+1

	print("number_correclyClassified: " + str(number_correclyClassified))
	print("len(valSet): "+ str(len(valSet)))
	ValError = (len(valSet)-number_correclyClassified)/len(valSet)

	return ValError

#Run Algorithm here
trainData = perceptron_train(trainFeatureVectors)
print(trainData)
validation = perceptron_validation(valFeatureVectors, trainData[0])
print("Validation Error: " + str(validation))


"""
Report

number of updates made(k): 76411
number of iterations: 1642

Validation Error: 0.059

"""








