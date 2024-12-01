# README - COSI 114b Logistic Regression Classifier

The code provided in this repository contains the solutions to the Logistic Regression Classifier PA for COSI 114b - Fundamentals of Natural Language Processing II. The assignment requires us to use Numpy and Scipy to implement the classifier. 

## Installation and Execution 

Get the files from GitHub and in your terminal/console move into the project folder. Run the file with the solutions with the following line: 

``` bash 
python logistic_regression.py 
```

Doing the above will run the code and use data given to the students. It is currently set to work on the larger files located in ``` movie_reviews ```, but can be changed to run on the smaller files, which is located in ``` movie_reviews_small ```. Since this code relies on the data files to work, make sure to unzip the compressed folders under the names mentioned before running the program. 

Note: These instructions assume that the user has python downloaded and is able to run the ``` python ``` command in their terminal. If they don't, they can either set their device up to be able to run the command, or they can open the ``` logisitic_regression.py ``` file in a separate application and run it through there. 


## Assignment Description 

The task is to implement a logistic regression classifier by completing the following functions: 

* ``` make_dicts(self, train_set) ``` - Given a folder of training documents, fills in ``` self.class_dict ```, a dictionary translating between class names and indices. 
Optionally, we may use ``` self.feature ``` to translate between feature names and indices; this is not required, but is strongly recommended if using a lot of word count features. For other types of features (described later in the ``` features ``` function below), we don't have to use ``` self.feature_dict ```, as long as we consistently use a single index for each feature. 
In addition, this function should set the number of features in ``` self.n_features ``` (including all features whether included in ``` self.feature_dict ``` or not), and initialize the parameter vector ``` self.theta ```. If $|F|$ is the number of features, ``` self.theta ``` is initialized as a vector of zeros, of length $|F| + 1$. 
* ``` load_data(self, data_set) ``` - Given a folder of documents (training, development, or testing), returns a list of ``` filenames ```, and dictionaries of ``` classes ``` and ``` documents ``` such that: 
    * ``` classes[filename] ``` = class of the document 
    * ``` documents[filename] ``` = feature vector for the document 
It may be helpful to store the classes in terms of their indices, rather than their names. ``` self.class_dict ``` can be used to translate between them. To get the feature vector for a document, the ``` featurize ``` function described below can be used. 
* ``` featurize(self, document) ``` - Given a document (as a list of words), returns a feature vector. Note that letting $|F|$ be the number of features, this function returns a vector of length $|F| + 1$. Furthermore, the last element of the vector should always be 1. If we consider our parameter vector ``` self.theta ``` to have form \begin{matrix} w_{1} & \cdots & w_{n} & b \end{matrix}, and our feature vector to have form \begin{pmatrix} x_{1} & \cdots & x_{n} & 1 \end{pmatrix}, then we can see that: 
```math
\begin{bmatrix}
x_{1} & \cdots & x_{n} & 1
\end{bmatrix}
\cdot 
\begin{bmatrix}
w_{1} \cr
\vdots \cr
w_{1} \cr
b \cr
\end{bmatrix} 
= 
\sum\limits_{j=1}^{n} x_{j} w_{j} + 1 \times b =  x \cdot w + b 
``` 
In this way, the last element of the vector, corresponding to the bias, is a "dummy feature" with value 1. 
What features should you use? You can start with word count features, as in the previous assignments. However, you are not limited to word count features. 
I used the binary featire so that loss could e calculated properly. The indices of the words of the vector are mapped to the valued indicated in ``` self.feature_dict ```. 
* ``` train(self, train_set, batch_size=3, n_epochs=1, eta=0.1) ``` - Given a folder of training documents, this function loads the dataset (using the ``` load_data ``` function), splits the data into mini-batches (``` minibatch ``` being a list of filenames in the current mini-batch), and shuffles the data after every eposh; these tasks have already been done for you. Your task is to implement the following for each mini-batch: 
1. Create and fill in a matrix **x** and a vector **y**. Letting *m* be the number of documents in the mini-batch, **x** should have shape ($m$, $|F| + 1$) and **y** should have shape ($m-{1}$). Inuitively, **x** consists of the feature vectors for each document in the mini-batch, stacked on top of each other, and **y** contains the classes for those documents, in the same order. In other words, **x** and **y** should have the following forms: 
```math
x = 
\begin{bmatrix}
x_{1}^{(1)} & \cdots & x_{n}^{(1)} & 1 \cr
\vdots & \ddots & \vdots & \vdots \cr 
x_{1}^{(m)} & \cdots & x_{n}^{(m)} & 1 \cr 
\end{bmatrix} 
y = 
\begin{bmatrix}
y^{(1)} \cr 
\vdots \cr  
y^{(1)} \cr  
\end{bmatrix} 
``` 
2. Compute $\hat{y} = \sigma(x \cdot \theta)$. Note the order of the operands $x \cdot \theta$ (rather than $\theta \cdot x$). 
3. Update the cross-entropy loss. Recall that the loss (for a single example) is $L_{CE}(\hat{y}, y) = -[y log \hat{y} + (1- y) log(1-\hat{y})]$. We want to calculate the average train loss over the whole training set, but since we're dividing by the number of training documents at the end, it's okay to just keep a running sum. 
4. Compute the average gradient $\nabla L =  \frac{1}{m}(x^{T} \cdot (\hat{y} - y))$. This equation computes the sum of the gradients for each document **i** in the mini-batch. To get the average gradient, we divide by **m**, the number of documents in the mini-batch. 
5. Update the weights (and bias). We can use the equation $\theta _{t+1} = \theta _{t} - \eta \nabla L$. 
As a general note, we were to avoid unnecessary for loops by taking advantage of Numpy *universal* functions, which automatically operate element-wise over arrays. These include arithmetic operations such as -, *, and ``` numpy.log ```, as well as functions that use them, like ``` scipy.special.expit ```. The last four steps should be done in one line of code each. 
* ``` test(self, dev_set) ``` - Given a folder of development (or testing) documents, returns a dictionary of ``` results ``` such that: 
    * ``` results[filename]['correct'] ``` = correct class 
    * ``` results[filename]['predicted'] ``` = predicted class 
We were free to store the classes in terms of their indices, rather than their names. We were also free to process the development/testing documents one at a time, rather than in mini-batches. Recall that $P(y = 1|x) = \hat{y}$, and we are using 0.5 as the decision boundary. 
* ``` evaluate(self, results) ``` - Given the result of ``` test ```, computes precision, recall, and F1 score for each class, as well as the overall accuracy, and prints them in a readable format. 
