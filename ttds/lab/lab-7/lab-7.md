# TTDS: Lab 7

Based on lectures 16 and 17

In this lab you will improve a text classification model for classifying tweets into 14 different categories.

## UNDERSTANDING A TEXT CLASSIFIER

- Download the following [**file**](https://opencourse.inf.ed.ac.uk/sites/default/files/https/opencourse.inf.ed.ac.uk/ttds/2023/tweetsclassification.zip)   
  There are two files in the compressed file:   
  - Tweets.14cat.train contains 2504 tweets to be used for training the classification model   
  - Tweets.14cat.test contains 625 tweets to be used testing   
  The format of the files is as follows (tab separated):   
      tweet_ID tweet category  
   
- Download the [Jupyter notebook](https://uoe-my.sharepoint.com/:u:/g/personal/bross3_ed_ac_uk/EX9dVD2pxFlGgFxAlKh_Xu8B2EKAawYyke7pap0Eu5Wh5A?e=PHivWG) with the code
- The code contains functions to **extract the BOW features** from the training files. It does the following:   
  - It reads the tweets and preprocesses them. No stemming or stopping takes place at this stage   
  - It finds all the unique terms, and gives each of them a unique ID (starting from 0 to the number of terms)
- Then it creates a count matrix where each row is a document and each column corresponds to a word in the vocabulary.   
  - The value of element (i,j) is the count of the number of times the word j appears in document i.   
  - Since most documents will not contain many words from the total vocabulary, it uses a sparse matrix. This means that only the nonzero values are actually stored in memory.   
  - This is implemented using the [dictionary-of-keys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html#scipy.sparse.dok_matrix) sparse matrix from [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html).  
  - See the Examples section of the [dok_matrix page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html#scipy.sparse.dok_matrix) (and lecture 18) for more information.  
  - Let's call this matrix X.
- The codes also makes a mapping from the categories to numberic IDs. So the list of correct categories \['Gaming', 'Gaming', 'Entertainment', ...\] becomes \[0,0,1, ...\] where the same category is also mapped to the same number from 0 to 1 - number of categories.   
  - It uses this mapping to convert the category for each tweet to a number. We call this list of correct categories y.
- Once the training and test features are ready, you can now train a classifier and test the performance. The example code starts with an [SVM multiclass Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).   
  - First it instantiates a model: model = sklearn.svm.SVC(C=1000, kernel="linear")   
  - The C parameter could be optimized for better performance. However, for this stage, we just set it to 1000.   
  - Then we fit the model to our data: model.fit(X,y) where X and y are defined as above.   
  - To make predictions about tweets, we can convert them to the same format as X and run y_pred = model.predict(X).   
  - y_pred now contains a list of the predicted categories corresponding to the documents in X.
- We use the same module from the previous steps to convert the test file of tweets into features file as well, using the same format.   
  - We make sure the same words are mapped to the same IDs as used in the training data. So if "dog" has the ID 453 in the training data, it should have the same ID in the test.   
  - For terms in the test file that do not exist in the training data (does not have a corresponding feature ID, since it never appear in the training data), please either neglect these terms, or define a new feature ID that corresponds to OOV (out-of-vocabulary) terms.
- Finally, we can classify the tweets from the testing set. We do NOT train the model again (model.fit()), only make new predictions given the testing data as input to model.predict().

## EVALUATION

Make sure you understand the code in the notebook.

Your first task, once you have done this, is to build a module that has the following outputs given the list of predicted values (category ids) and the list of correct categories (category ids):

- Accuracy
- Precision, recall, and F1 for each of the classes
- Macro-F1 score of the system

The notebook calculates this using the classification_report function. By implementing the functions yourself and comparing the results, you can make sure your understanding of classifier evaluation is correct.

## MODIFYING YOUR CLASSIFIER

Next, think about what can make a better classification.

You can think about trying the following:

- Apply stemming and stopping.
- Duplicate hashtags words (e.g. convert "#car" to "#car car").
- Expand tweet text that has link with the page title of that link
- Try non-textual features: Tweet length, presence of hashtags, links, emojis, etc.
- Use Tf-IDF features intead of just counts.
- Can you try another classifier and compare the results?
- Get more training data! You can check [this paper](http://arxiv.org/pdf/1503.04424.pdf) for potential ideas on the same task
- Any other ideas you have about how to make the model better.

Try at least two possible methods for improving the classification.

- Find out which method achieves the best performance (measure by Macro-F1), and generate the confusion matrix among all classes for it.
- (Advanced, optional): Try a HuggingFace model for sequence classification, like here: https://huggingface.co/docs/transformers/tasks/sequence_classification

### NOTE

Parts of this lab are similar to CW2 (though on a different dataset), so please try to finish it properly.

You can share results and findings of what works well *for the lab data* with other students. Just make sure not to share anything about the coursework data.
