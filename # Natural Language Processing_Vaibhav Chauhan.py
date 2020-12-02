# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting=3)


import re
import nltk                                                 		 #nltk lib is used to download stopwords…which is used to…to remove words(basically it gives info which words to remove)
nltk.download('stopwords')                                  	 #stopwords - list of words which are irrelevant(ex -the,a,an)
from nltk.corpus import stopwords					#importing stopwords
from nltk.stem.porter import PorterStemmer                       #PorterStemmer is a class from nltk.stem.porter library used to apply stemming on reviews  &&… ps is object of this class
corpus=[]                                                     				#name of list(text)
for i in range(0,1000):								#1000-no of reviews
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   		 #only considering a-z &A-Z and replacing others(^) by space && Review is the name of the column
    review=review.lower()								#capital->lower case
    review=review.split()                                    												#to split the review(string) in diff words
    ps=PorterStemmer()                                                      										#stemming-taking root of words (no tense ex- loved->love)-to avoid too much sparsity(by decreasing no of columns)
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   			#this for loop is to go though diff word in a review 1)only consider words not in stop word 2)applying stemming on them.
    review=' '.join(review)                                                                    								#and using set to convert list of word to set because its fast(for long reviews)
    corpus.append(review)
    
                                                            													#bag of words-take all words of corpus and give each word one column, rows our reviews
                                                           													 #it is created by tokenization
#creating bag of words model(its a matrix which is very sparse)
from sklearn.feature_extraction.text import CountVectorizer			 #bag of words is a matrix with each word having 1 column
cv= CountVectorizer(max_features=1500)                					 #max columns =1500(1500 most common words)   we had an option to do the clean here in cv but we prefer doing it manually as it gives us freedom
X=cv.fit_transform(corpus).toarray()                 						 #creating sparse matrix,converting matrix to array  2-d               ....like in html page code, we have to remove br,etc.
y=dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()                                     						#classification model based on naive bayes
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix & accuracy
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)                						#got an accuracy of 78% on 200 reviews in test set , trained on 800
print(cm)
accuracy_score(y_test, y_pred)





—>On analysis I found that stopword removed not also ….which is not right so:-
	To remove ‘not’ from stopwords:-
		xx=stopwords.words('english')
		xx.remove(’not’)
	 	review=[ps.stem(word) for word in review if not word in set(xx)] 

—>actually the accuracy was 78 % on test set of 200 … if I use k-fold cross validation then the accuracy I got was around 81%….explain fold cross val.
		from sklearn.model_selection import cross_val_score
		accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)																#10 sets && mean of those 10 accuracy is printed
		print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

—>and after that I used grid search to improve the accuracy of my classifier by using grid search, which helps in tuning the hyper parameters  which helped in achieving accuracy around 82%:
		from sklearn.model_selection import GridSearchCV
		parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
		grid_search = GridSearchCV(estimator = classifier,
                 		          param_grid = parameters,
                         		  scoring = 'accuracy',
                           		cv = 10,
                           		n_jobs = -1)																											#njobs=-1 to use all cores of pc
		grid_search.fit(X_train, y_train)
		best_accuracy = grid_search.best_score_
		best_parameters = grid_search.best_params_
		print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
		print("Best Parameters:", best_parameters)



		