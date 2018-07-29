# CreditCardFraudDetection

INTRODUCTION:
In modern banking, the usage of credit cards has increased drastically due to the ease and benefits of using credit cards over other methods of payment along with the cases of fraud associated. The banks, merchants and credit card companies lose billions of dollars every year to credit card frauds. It is important for credit card companies to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The aim for this research is to develop a model that can predict the fraudulent transactions accurately to prevent unintended losses incurred to the customers.

Proposed statistical tests (classification models) to predict fraud using R programming language
	•	Decision Trees
	•	Random forest
	•	Logistic regression
	•	k-nn
	•	Support vector machines


Code: This directory contains R based *.Rmd code files containing code for the implementation of above project.

ProjectOutput: This folder has html files containing Data Description and Attribute Visualization and Classification models using Resampling Techniques. 

**Download the html files first to your desktop to view them properly.

Kirandeep Kaur Kandhola
Ryerson Id: 500866930
Credit Card Fraud Detection

INTRODUCTION:
In modern banking, the usage of credit cards has increased drastically due to the ease and benefits of using credit cards over other methods of payment along with the cases of fraud associated. The banks, merchants and credit card companies lose billions of dollars every year to credit card frauds. It is important for credit card companies to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The aim for this research is to develop a model that can predict the fraudulent transactions accurately to prevent unintended losses incurred to the customers.

Proposed statistical tests (classification models) to predict fraud using R programming language
	Decision Trees
	Random forest
	Logistic regression
	k-nn
	Support vector machines


LITERATURE REVIEW:
	Boughorbel, S., Fethi, J. and Mohammed, E. Optimal Classifier for Imbalanced Data Using Matthews Correlation Coefficient Metric. PLoS ONE, 12, e0177678, (2017).
This article gives insight about Area Under Receiver Operating Characteristics Curve (AUC) and Matthews Correlation Coefficient (MCC). These are widely used metrics of success designed to handle data imbalance. The limitation of using AUC is that there is no explicit formula to compute AUC. On the other hand, MCC has a close form and it is very well suited to be used for building the optimal classifier for imbalanced data.
The chosen Credit card fraud detection dataset is highly imbalanced, so AUC and MCC can be used to get fairly robust results which are not much affected by the imbalance of the class.

	N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002. 
This paper gives an insight to deal with imbalanced data using the combination of  over-sampling the minority (abnormal) class and under-sampling the majority (normal) class which achieve better classifier performance (in ROC space) than only under-sampling the majority class. Under- sampling of majority class increase the sensitivity of a classifier to the minority class. SMOTE was proposed as an over-sampling approach in which the minority class is over-sampled by creating “synthetic” examples along the line segments joining any/all of the k minority class nearest neighbors rather than by over-sampling with replacement. 
All these methods to deal with skewed data can used to evaluate the classifier in ROC space for my dataset and then comparison can be done between these methods for the various classifiers used.

	K. R. Seeja and Masoumeh Zareapoor, “FraudMiner: A Novel Credit Card Fraud Detection Model Based on Frequent Itemset Mining,” The Scientific World Journal, vol. 2014, Article ID 252797, 10 pages, 2014. https://doi.org/10.1155/2014/252797.
This paper proposes using frequent itemset mining model for detecting fraud from highly imbalanced and anonymous credit card transaction datasets. An algorithm is also proposed to find to which pattern (genuine or fraud) the incoming transaction of a particular customer is closer and a decision is made accordingly. In order to handle the anonymous nature of the data, no preference is given to any of the attributes and each attribute is considered equally for finding the patterns. This paper also gives insight to various classifiers used for binary classification and sensitivity (fraud detection rate), false alarm rate, balanced Classification Rate (BCR) and Matthews Correlation Coefficient (MCC).
The chosen credit card fraud detection dataset has PCA transformed features; it will be interesting to create separate genuine transaction pattern (costumer buying behavior pattern) and fraud transaction pattern (fraudster behavior pattern) and to find if the incoming transaction is closer to genuine/fraudulent transaction pattern.
	L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001. 
This deals with Random forest which is an ensemble of decision trees. The basic principle behind is that a group of “weak learners” can come together to form a “strong learner.” Random forests grow many decision trees. Here each individual decision tree is a “weak learner,” while all the decision trees taken together are a “strong learner.” When a new object is to be classified, it is run down in each of the trees in the forest. Each tree gives a classification output or “vote” for a class and classifies the new object into the class having maximum votes. Random forests are fast and they can efficiently handle unbalanced and large databases with thousands of features. 
Thus it can effectively and efficiently handle the given credit card fraud detection dataset having 284,807 transactions with only 0.172 % fraudulent transactions.

	S. Maes, K. Tuyls, B. Vanschoenwinkel, and B. Manderick, “Credit card fraud detection using Bayesian and neural networks,” in Proceedings of the 1st International NAISO Congress on Neuro Fuzzy Technologies, pp. 261–270, 1993.
This research paper deals with artificial neural networks and Bayesian networks. These methods provide computational learner with a set of training data. After the process of learning, the program is supposed to be able to correctly classify a transaction it has never seen before as fraudulent or genuine, given some features of that transaction and it can be effectively applied to credit card fraud detection with anonymous data with PCA transformed features. 
This paper also deals with different problems in credit card fraud detection like skewness of the data, need of good metric to evaluate the classifier system which also happens to be the case in my research topic where fraudulent transactions are 0.172 %. The classifier should be able to adapt themselves to a new kind of frauds. Since a while successful fraud techniques decrease in efficiency, due to the fact that they become well known.

	E. Duman and M. H. Ozcelik, “Detecting credit card fraud by genetic algorithm and scatter search,” Expert Systems with Applications, vol. 38, no. 10, pp. 13057–13063, 2011.
This research paper deals with Genetic algorithms: these are inspired from natural evolution. The basic idea is that the survival chance of stronger members of a population is larger than that of the weaker members and as the generations evolve the average fitness of the population gets better. Normally the new generations will be produced by the crossover of two parent members. It starts with a number of initial solutions which act as the parents of the current generation. New solutions are generated from these solutions by the cross-over and mutation operators. The less fit members of this generation are eliminated and the fitter members are selected as the parents for the next generation. This procedure is repeated until a pre-specified number of generations have passed, and the best solution found until then is selected. 
Genetic Algorithm generate better solutions as time progresses, thus can be used in my research. Detecting a fraud on a card having a large available limit is more valuable than detecting a fraud on a card having a small available limit. Classical DM algorithms are not designed for such misclassification cost structure, thus as an alternative to classical DM algorithm it will be interesting to see how GA performs on the given anonymous credit card fraud detection dataset.

	P. K. Chan, W. Fan, A. L. Prodromidis, and S. J. Stolfo, “Distributed data mining in credit card fraud detection,” IEEE Intelligent Systems and Their Applications, vol. 14, no. 6, pp. 67–74, 1999. 
In this paper a large data set of labeled transactions is divided (either fraudulent or genuine) into smaller subsets, mining techniques are applied to generate classifiers in parallel, and combined the resultant “base" models by “meta-learning" from the classifiers' behavior to generate a “meta-classifier." 
This method can deal with problems of my data set. First, there are millions of credit card transactions processed each day so the model can be easily scaled if has to be used in real time. Second, it deals with highly skewed data effectively and third, as each transaction record has a different dollar amount and hence has a variable potential loss, rather than a fixed misclassification cost per error type, as is commonly assumed in “cost-based" mining techniques.

	S..Dhok, G.R. Bamnote, “Credit Card Fraud Detection Using Hidden Markov Model”, International Journal of Advanced Research in Computer Science, Volume 3, No. 3, May-June 2012.
This research paper deals with detecting fraud using Hidden Markov model. An HMM is initially trained with the normal behavior of a cardholder. If an incoming credit card transaction is not accepted by the trained HMM with sufficiently high probability, it is considered to be fraudulent. Hidden Markov Model helps to obtain a high fraud coverage combined with a low false alarm rate which can be taken as metric of success in case of highly imbalanced data as is the case with my chosen dataset.
	C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.
This research deals with Support Vector Machines which are statistical leaning techniques for two-group classification problems. SVM algorithms tend to construct a hyperplane as the decision plane which does separate the samples into the two classes—positive and negative. This algorithm finds a special kind of linear model, the maximum margin hyperplane, and it classifies all training instances correctly by separating them into correct classes through a hyperplane. The maximum margin hyperplane is the one that gives the greatest separation between the classes. The instances that are nearest to the maximum margin hyperplane are called support vectors. There is always at least one support vector for each class, and often there are more. 
In credit card fraud detection, it determines if the test instance falls within the learned region. Then if a test instance falls within the learned region, it is declared as normal; else it is declared as anomalous (fraudulent). This model has been demonstrated to possess a higher fraud detection efficiency of credit card fraud detection compared with other algorithms and larger training datasets are an asset to achieve maximum prediction accuracy.


Dataset:
The dataset has been taken from kaggle competitions          https://www.kaggle.com/mlg-ulb/creditcardfraud/data 
It contains transactions made by credit card in two days in September 2013 by European cardholders. Due to confidentiality issues; the original features and more background information about the data are not provided, thus the transactions are completely anonymous.
	Features V1, V2 ... V28 are the principal components obtained with PCA.
	'Time' and 'Amount' features are not transformed with PCA. 'Time' contains the seconds elapsed between the given transaction and the first transaction in the dataset, thus it acts like an index. 'Amount' here is the transaction Amount. 

	Data Types:	
All the attributes including PCA transformed features V1 to V28, Time, and Amount are all numeric. However the data type of Class is integer that can be changed to factor (takes value of 1 in case of fraud and 0 in case of genuine transaction). This is the column we would like our model to learn from and be able to predict frauds for new transactions.
1.2	Missing Values
There are no missing values in data provided. V1 through V28 (principal components of the PCA), Time and Amount also doesn’t have any missing value.

1.3	Summary Statistics
 
	All the PCA transformed anonymous attributes have a mean of zero and seems to be normalized.
	Time keeps on increasing as is evident from the description of this attribute with a min value of 0 and a maximum value of 172792.
	Amount varies from 0 to 25691.16 and there are no transactions with negative values. This attribute will be transformed to facilitate training ML models.
	There are 492 frauds out of 284,807 transactions, which accounts for 0.172% of all transactions leaving the data highly imbalanced.


1.4	Skewness of the data
 
	Although features from V1 to V28 are PCA transformed and all have a mean of 0, still V1, V2, V8, V17, V23 are negatively skewed and  V21, V28 are positively skewed. 
	Amount is heavily skewed to the right with a different range of values than other variables. So, the amount will be normalized.
	Outliers are present in the Amount, PCA transformed features were found using 
outlier_values <- boxplot.stats(ccfd$V8)$out  but decided to preserve them as the features are already PCA transformed.


1.5	Visualization of Class Imbalance
  		
As is clear, the data is highly imbalanced. Also, the levels of the class are reordered to fit the Fraud Class.


1.6	Duplicate Values
	V1 to V28 are PCA transformed, still checked for duplicates keeping all the features: Time, V1:V28, Amount, and Class. There are 1081 duplicates and found that almost 1.757 % of duplicate transactions belong to fraudulent Class which is 10 times more than fraud cases found in total transactions provided.
		 
	Excluding “Time” and “Class” to find duplicates gives 9144 duplicated records, thus for 9144 times, a credit card has been used for same transaction amount under similar circumstances but at different time thus  it might be possible that once it was classified as fraud and other time as genuine. 
	Excluding only “Time” gives same no of duplicates (9144) as are found by excluding “Time” and “Class” both, thus a credit card could have been used for same transaction amount under similar circumstances but at different time and have the same class i.e. fraud or genuine, which is quite possible. 




1.7 	Correlations between the Attributes and the “Class”
 
	It is clear that most of the data features are not correlated to each other because before publishing, real features were transformed through PCA. We do not know if the numbering of the features reflects the importance of the Principal Components.
	The “Class” is not significantly related to V19 to V28.
	Amount of Transaction is negatively correlated to V5, V2, V1, and V3, and positively correlated to V7, V6, V20.
	As Time indicates the time elapsed between the first transaction and the given transaction, so it doesn’t bear much meaning.


1.8	Visualization of attributes 
1.8.1	Visualization of all the features for fraudulent and genuine transactions
For a classifier to detect fraudulent transactions from genuine, we expect that the distribution of variables for genuine transactions is different from the distribution for fraudulent ones. Some plots are made to verify this.
 
	The distributions for fraudulent transactions are very different than genuine, except for the Time feature, which seems to have the exact same distribution.
	For features V13, V15, V19 to V26 , the distributions for fraudulent and genuine transactions are not distinguishable.
1.8.2	Visualization of mean values of all the features for fraudulent and genuine transactions
Find the mean values of all the features for both genuine and fraudulent transactions and plot their variations. 
	The mean value of some of the features for anomalous examples falls out of range. 
	Some features (19 through 28) do not vary as much. Hence, these features can be safely ignored and focus on the features those contribute significantly for a transaction to be identified as fraud. 

Approach
 

	The dataset contains 30 numerical input variables: Time, Amount, and V1 to V28 are the result of PCA transformation.  Target variable was changed to factor and levels were reordered to fit the “Fraud” Class.

	Performance Metrics: A performance metric is selected to evaluate the performance of Classifier models.
Confusion Matrix Terminology:

Prediction 
Actual
	Fraud(Positive)	Genuine(Negative)
Fraud(Positive)	TP	FP
Genuine(Negative)	FN	TN

True Positive Rate/Sensitivity/Recall: It refers to fraud catching rate (should be high) and represents the portion of actual positives which are predicted positive.   TP⁄((TP+FN))

False Positive Rate/ False Alarm Rate: It represents the portion of actual negatives which are predicted as positive and this metric should be low as false alarm leads to customer dissatisfaction.     FP⁄((FP+TN))

Precision (Positive Predictive Value): It is a measure of correctness and represents the ratio of actual positives out of the total predicted positives (Frauds).     TP⁄((TP+FP))

Specificity/ True Negative Rate: It measures the proportion of actual negatives that are correctly identified as negative.	TN⁄((TN+FP))
  
F- Score (weighted average of precision recall): It is a measure that combines precision and recall and is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:   (2*Prec*Recall)⁄((Prec+Recall))

Accuracy: This metric tells how often a classifier is correct.	((TP+TN))⁄((TP+TN+FN+FP))
In case of unbalanced data, accuracy cannot be used to evaluate the performance of the model because it tend to fit the majority class better than the minority class, thus a classifier can achieves an accuracy of approx. 99.8 %, if it classifies all instances as the majority class and eliminates the 0.172 % minority class observations as noise. Thus preferred metrics for evaluation of the model is Area Under ROC Curve (AUC).

Receiver Operating Characteristics (ROC) curve is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) with varying threshold for assigning observations to a given class and the area under the curve (AUC) is taken as evaluation measure of the classifier [1].

	Features selection: 
Feature 'Time' contains the seconds elapsed between the given transaction and the first transaction in the dataset. Since we do not have a begin time for the first transaction, the time column contains very little useful information, moreover it has same distribution for fraudulent and genuine transactions (1.8.1) so the ‘Time’ feature is removed from data.
A set of features was selected using Recursive feature elimination, Ranking features by importance with “glm”, “rpart”, “C5.0” functions and by step forward selection, step backward selection method.  leaps::regsubsets is also used to select best subsets.

Remove Redundant Features: All the features are PCA transformed thus are not correlated to each other. None of them correlation of 0.75 or higher.
Recursive feature elimination: It suggests that taking all the attributes into consideration will give better results and also that there should not be much difference in the results if we choose a subset of 20 or 25 or 30 attributes.
Step forward and backward selection suggests taking all the features into consideration, according to difference in the AIC value of all the models.
Top 20 features using “glm”:
V4, V14, V10, V27, V21, V8, V20, V22, 13, V28, Amount, V9, V1, V7, V6, V23, V5, V16, V15, V24
Top 20 features using “rpart”:
V17, V12, V14, V10, V16, V9, V7, V3, V2, V11, V4, V18, V5, V26, V15, V21, V22, V23, V27, V24 
Top 20 features using “C5.0”:
V10, V4, V14, V17, V9, V18, V12, V26, V6, V7, V20, V23, V22, V19, V11, V3, V15, V16, V8, V1 
Best subset of 8 features using regsubsets:	      V17, V14, V12, V10, V16, V3, V7, V11


Top 5 features of all the methods were selected (to avoid biasing towards one classifier) which include V4, V8, V10, V12, V14, V16, V17, V21, V27, and Amount. Performance of the classifier is compared using all the features and the selected features in the results section.

GIT Path: https://github.com/kaurkandhola/CreditCardFraud.git
Branch: master
File: Code/Preprocessing and Visualization.Rmd

	Training and Test Datasets:
The dataset is split into 80% training and 20% testing maintaining the class ratio of the original dataset. To preserve the ratio of the target factors, caret::createPartition is used for stratified split of the data into training and testing.
Normalization:
The training and test datasets are normalized separately to avoid information leak about the testing data before the split.

	Repeated Cross Validation: 
The classifier is trained on the normalized training data using repeated 10 fold cross validation with 3 repeats and caret:: createFolds is used for stratified splitting of the data in the folds.

	Resampling Methods:
It consists of removing samples from the majority class and or adding more examples from the minority class. In case of highly imbalanced data, resampling is done to balance the classes, as most of the machine learning Classifiers work to give better accuracy and the class imbalance will affect the performance. Following are the different resampling methods [2]:
	Random Under-Sampling: Random Under Sampling aims to balance class distribution by randomly eliminating majority class examples and until class instances are balanced out.
	Random Over-Sampling: Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.
	Informed Over Sampling (SMOTE): Synthetic Minority Over-sampling Technique: This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.
	Cost sensitive learning: Cost-Sensitive Learning is a type of learning in data mining that takes the misclassification costs (and possibly other types of cost) into consideration. The goal of this type of learning is to minimize the total cost.

GIT Path: https://github.com/kaurkandhola/CreditCardFraud.git
Branch: master
File: Code/DT with Resampling Methods.Rmd


	Base Model:
Decision Tree (caret::rpart) is chosen as base model to compare the AUC using different resampling methods of and its performance is discussed in the results section.



	Classification Models :
This section provides the details of the algorithms used in the experiment and  the results section provides results of these algorithms and discuss their usefulness in fraud detection.

Decision Trees:  
Decision tree builds non-linear classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node has two or more branches whereas a leaf node represents a classification or decision. The topmost decision node in a tree which corresponds to the best predictor called root node which has no incoming edges. Decision trees can handle both categorical and numerical data. 
For the credit card dataset, caret::rpart is used to evaluate the performance [3].

Random Forest:
Random Forest is a combination of many decision trees. The basic principle behind ensemble methods is that a group of “weak learners” can come together to form a “strong learner”. Random forest grows many trees creating N datasets by randomly sampling with replacement from the original training dataset and only a subset of attributes are selected randomly. The criterion of information gain is then applied for the selection of the best attributes to that sample of attributes. When a new instance is to be classified, it is run down in each of the trees in the forest. Each tree gives a classification output or “vote” for a class. The forest classifies the new object into the class having maximum votes. Random forests are fast and they can efficiently handle unbalanced and large databases with thousands of features [4].
Logistic Regression:
 Logistic regression, or logit regression, is a probabilistic statistical classification model which determines the relationship between continuous or categorical feature variables with the binary class outcome of the instance. Logistic model is a Classifier where the log-odds of the probability of an event is a linear combination of independent or predictor variables and it doesn’t suffer from severe class imbalance thus it show nearly same performance by using imbalanced dataset and by using smote [5].

K-NearestNeighbor:
k-nn is a non-linear classifier that stores all available instances and then classify new instances based on a similarity measure. This similarity measure is generally considered to be a distance function (Euclidian distance or Minkowski distance or Manhattan distance). In k-nn algorithm, an instance is classified by a majority vote of its neighbors with the case being assigned to the class most common amongst its k nearest neighbors using a distance metric. As distance functions are used to show the similarities, it is very important to have normalization of the features. Sometimes more than one nearest neighbor is used, and the majority class of the closest K neighbors is assigned to the new instance. Thus we classify any incoming transaction by calculating preset nearest points to new incoming transaction, if majority of nearest neighbors are fraudulent, then the transaction is classified as fraudulent and if majority of nearest neighbors are genuine, then is classified as genuine [6].

Support Vector Machine:
The support vector machine (SVM) is a discriminative classifier based on the concept of decision planes which define decision boundaries. The algorithm outputs an optimal hyperplane(decision plane) which categorizes new examples when provided with labeled training data.  The strength of SVMs comes from two main properties: kernel representation and margin optimization. Kernels, such as radial basis function (RBF) kernel, can be used to learn complex regions and Extends to patterns that are not linearly separable by transformations of original data to map into new space to find a special kind of linear model, the maximum margin hyperplane, and it classifies all training instances correctly by separating them through a hyperplane. The maximum margin hyperplane is the one that gives the greatest separation between the classes. The instances that are nearest to the maximum margin hyperplane are called support vectors. There is always at least one support vector for each class, and often there are more. SVM methods require large training dataset sizes in order to achieve their maximum prediction accuracy [7].

GIT Path: https://github.com/kaurkandhola/CreditCardFraud.git
Branch: master
File: Code/Classifiers with SMOTE.Rmd



Results:
Total number of records (instances): 284315
     Fraud: 492
Genuine:  284315
TABLE1:	For stratified 80% - 20 % split 
	Training set :            	Test Set
Instances for split	227846	56961
Fraud	394	98
Genuine	227452	56863

Baseline Model: 
Decision tree Classifier is used as baseline model with AUC value of 0.8825. Different classifiers will be used with tuning parameters with the goal of increasing the value of chosen performance metric i.e. AUC value. 
AUC for baseline model: 0.8825

TABLE 2:	Comparison of different resampling methods using caret::rpart
Resampling method	Confusion Matrix	Sensitivity	ROC* curve	AUC
Unbalanced	Pred
	Actual
	Frd	G
Frd	79	17
G	19	56846
	0.806	 	.9089
Smote	Pred
	Actual
	Frd	G
Frd	89	2519
G	9	54344
	0.908	 	0.9485
Up	Pred
	Actual
	Frd	G
Frd	86	2518
G	12	54345
	0.877	 	0.9395
Down	Pred
	Actual
	Frd	G
Frd	88	2879
G	10	53984
	0.898	 	0.9478
(*ROC is plotted with 20 threshold values)
From the results, it is seen that the sensitivity (Fraud catching rate) is the highest for the classifier using smote resampling. Also the classifier gives best AUC using smote resampling, thus we will use smote resampling to compare with other classifiers.

TABLE 3:	Comparison of caret::rpart classifier using different features using smote resampling

	Confusion Matrix	Sensitivity	ROC Curve	AUC
All Features	Pred
	Actual
	Frd	G
Frd	89	2519
G	9	54344
	0.908	 	0.9485
10 Selected  Features	Pred
	Actual
	Frd	G
Frd	87	2843
G	11	54020
	0.887	 	0.9464
Results above show that selecting all the features increases AUC and sensitivity both, so we will compare different classifiers using all features with smote resampling.































TABLE 4:	Comparison of classifiers using all features with smote resampling
	Confusion Matrix	Sensitivity	ROC curve	AUC
Rpart
(cp=0)


	Pred
	Actual
	Frd	G
Frd	89	2519
G	9	54344
	0.9080	 	0.9485
Random forest
(mtry =8)


	Pred
	Actual
	Frd	G
Frd	88	532
G	10	56331
	0.8979	 	0.9857
Logistic regression


	Pred
	Actual
	Frd	G
Frd	89	1226
G	9	55637
	0.9081	 	0.9759
k-nn 
(k=21)


	Pred
	Actual
	Frd	G
Frd	87	841
G	11	56022
	0.8875	 	0.9737
SVM
(sigma = 0.0321,
C = 0.5)

	Pred
	Actual
	Frd	G
Frd	87	1522
G	11	55341
	0.8877	 	0.9822
*tuning parameters are given in the parenthesis





ROC curve for classifiers:
 


Conclusions:
 In this capstone research project, I have analysed anonymous credit card fraud detection dataset. Five different classification techniques (Decision Tree, Random Forest, Logistic regression, k-nn, SVM) are considered to find a classifier that can detect frauds effectively, taking into account the customer dissatisfaction that from contacting customers (calls, questionnaire) before a transaction. SMOTE is applied to avoid biasing of a classifier to fit one class over the other. Decision tree has the lowest AUC than all other classifiers. k-nn has AUC more than decision tree but less than Random Forest, Logistic Regression and SVM. In terms of sensitivity, Logistic Regression has highest value but the AUC value is lower than SVM and Random Forest. SVM has AUC value of 0.9822 with fraud catching rate of 88.77% and false alarm rate of 2.6% which can lead to customer dissatisfaction. Random Forest  has AUC value of 0.9857 with fraud catching rate of 89.79% (higher than SVM) and false alarm rate of 0.9% which is quite good in comparison to SVM. 
So I would suggest using Random Forest to detect fraud. As both customer and fraudulent behaviors are found to be changing gradually over a longer period of time, this may degrade the performance of fraud detection model. These behavioral changes can be incorporated into the proposed model by updating the fraud and geniune pattern databases. This can be done by running the proposed classification algorithim at fixed time points like once in 3 months or six months or once in every one lakh transaction.


References:
[1]	https://en.wikipedia.org/wiki/Receiver_operating_characteristic
[2]	https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
[3]	http://www.saedsayad.com/decision_tree.htm
[4]	L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001
[5] 	Dr.Ceni Babaoglu, Logistic Regression,Lecture 8,CMTH 642
[6] 	https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
[7] 	C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

