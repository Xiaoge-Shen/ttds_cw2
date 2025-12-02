<font style="color:rgb(54, 54, 54);">Based on lectures 8-9, 12-13, 16-17</font>

<font style="color:rgb(54, 54, 54);">This coursework is split into three main parts:</font>

1. **<font style="color:rgb(54, 54, 54);">IR Evaluation</font>**<font style="color:rgb(54, 54, 54);">: based on lectures 8 and 9</font>
2. **<font style="color:rgb(54, 54, 54);">Text Analysis</font>**<font style="color:rgb(54, 54, 54);">: based on lectures 12/13 and lab 6</font>
3. **<font style="color:rgb(54, 54, 54);">Text Classification</font>**<font style="color:rgb(54, 54, 54);">: based on lectures 16/17 and lab 7</font>

## <font style="color:rgb(54, 54, 54);">Important dates</font>
### <font style="color:rgb(54, 54, 54);">Submission Deadline: Friday, 28 November 2025, 12:00 PM (noon)</font>
## <font style="color:rgb(54, 54, 54);">1. IR Evaluation</font>
<font style="color:rgb(54, 54, 54);">In the first part of the coursework, you are required to build a module to evaluate IR systems using different retrieval scores. The inputs to your module are:</font>

1. <font style="color:rgb(54, 54, 54);">system_results.csv: a file containing the retrieval results of a given IR system and</font>
2. <font style="color:rgb(54, 54, 54);">qrels.csv: a file that has the list of relevant documents for each of the queries.</font>

<font style="color:rgb(54, 54, 54);">Please follow these steps:</font>

+ <font style="color:rgb(54, 54, 54);">Download the following files:</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">system_results.csv</font>](https://opencourse.inf.ed.ac.uk/sites/default/files/2023/ttdssystemresults.csv)<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">and</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">qrels.csv</font>](https://opencourse.inf.ed.ac.uk/sites/default/files/2023/qrels.csv)<font style="color:rgb(54, 54, 54);">. More about the 2 files:   
</font><font style="color:rgb(54, 54, 54);">- system_results.csv file, containing the retrieved set of documents for 10 queries numbered from 1 to 10 for each of 6 different IR systems. The format of the files is as follow: </font>

```plain
system_number,query_number,doc_number,rank_of_doc,score 
    1,1,710,1,5.34
    1,1,213,2,4.23
    1,2,103,1,6.21
```

<font style="color:rgb(54, 54, 54);">- qrels.csv file, which contains the list of relevant documents for each of the 10 queries. The format of the file is as follows: </font>

```plain
query_id,doc_id,relevance
    1,9090,3
    1,6850,2
    1,9574,2
```

<font style="color:rgb(54, 54, 54);">where the first number is the query number, the second is the document number and the third is the relevance. e.g. 1,9090,3 means that for query 1, document 9090 has a relevance value of 3. This value is only important for measures such as DCG and nDCG; while for measures such as P, R, and AP, all listed documents as relevant are treated the same regardless to the value.</font>

+ <font style="color:rgb(54, 54, 54);">Develop a module EVAL that calculates the following measures:   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">P@10</font>**<font style="color:rgb(54, 54, 54);">: precision at cutoff 10 (only top 10 retrieved documents in the list are considered for each query).   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">R@50</font>**<font style="color:rgb(54, 54, 54);">: recall at cutoff 50.   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">r-precision</font>**<font style="color:rgb(54, 54, 54);">   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">AP</font>**<font style="color:rgb(54, 54, 54);">: average precision   
</font>_<font style="color:rgb(54, 54, 54);">hint</font>_<font style="color:rgb(54, 54, 54);">: for all previous scores, the value of relevance should be considered as 1. Being 1, 2, or 3 should not make a difference on the score.   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">nDCG@10</font>**<font style="color:rgb(54, 54, 54);">: normalized discount cumulative gain at cutoff 10.   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">nDCG@20</font>**<font style="color:rgb(54, 54, 54);">: normalized discount cumulative gain at cutoff 20.   
</font>_<font style="color:rgb(54, 54, 54);">Note</font>_<font style="color:rgb(54, 54, 54);">: Please use the equation in</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">Lecture 9</font>](https://opencourse.inf.ed.ac.uk/sites/default/files/2024-10/ttds24_09evaluation.pdf)<font style="color:rgb(54, 54, 54);">. Any other implementation for nDCG will not be accepted.</font>
+ <font style="color:rgb(54, 54, 54);">The following file needs to be created in the exact described format.   
</font><font style="color:rgb(54, 54, 54);">-</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">ir_eval.csv</font>**<font style="color:rgb(54, 54, 54);">: A comma-separated-values file with the format:</font>

```plain
system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20
    1,1,0.00,0.00,0.00,0.00,0.00,0.00
    1,2,0.00,0.00,0.00,0.00,0.00,0.00
    ...
    1,10,0.00,0.00,0.00,0.00,0.00,0.00
    1,mean,0.00,0.00,0.00,0.00,0.00,0.00
    2,1,0.00,0.00,0.00,0.00,0.00,0.00
    2,2,0.00,0.00,0.00,0.00,0.00,0.00
    ...
    6,10,0.00,0.00,0.00,0.00,0.00,0.00
    6,mean,0.00,0.00,0.00,0.00,0.00,0.00
```

<font style="color:rgb(54, 54, 54);">that includes the evaluation results for the 6 systems, labelled with their system_number (1-6) which matches the number from results.csv.   
</font><font style="color:rgb(54, 54, 54);">- Each row should contain a list of the above scores for one of the systems for one of the 10 queries. A full example output file (with, incorrectly, all scores as 0) can also be found</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">here</font>](https://opencourse.inf.ed.ac.uk/sites/default/files/2023/irevalexample.csv)<font style="color:rgb(54, 54, 54);">.   
</font><font style="color:rgb(54, 54, 54);">- For each system, You should also include a row with the query_number set to "mean" for each system that includes the average results for each metric for that system across all 10 queries.   
</font><font style="color:rgb(54, 54, 54);">- Before submission, please check that your out files for this file is correct using the</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">Python script</font>](https://uoe-my.sharepoint.com/:u:/g/personal/bross3_ed_ac_uk/ETOMnSjjkjJNhfFJEChlAiEB_otDO0AadDFA0uLCVMo2kg?e=wMPvd5)<font style="color:rgb(54, 54, 54);">.</font>

+ <font style="color:rgb(54, 54, 54);">Based on the average scores achieved for each system, add a section in your report called "IR Evaluation" to describe the best system according to each score (i.e. what is the best system when evaluated using with P@10, and what is the best system with R@50, and so on). For each best system with a given score, please indicate if this system is statistically significantly better than the second system with that score or not. Please explain why.   
</font>_<font style="color:rgb(54, 54, 54);">hint</font>_<font style="color:rgb(54, 54, 54);">: using 2-tailed t-test, with p-value of 0.05. You are free to use existing tool for calculate the p-value. No need to implement this one.</font>
+ **<font style="color:rgb(54, 54, 54);">NOTE</font>**<font style="color:rgb(54, 54, 54);">:   
</font><font style="color:rgb(54, 54, 54);">- All files of results will be marked automatically. Therefore, please be careful with using the correct format.   
</font><font style="color:rgb(54, 54, 54);">- Please round the scores to 3 decimal points (e.g.: 0.046317 --> 0.046).</font>

## <font style="color:rgb(54, 54, 54);">2. Text Analysis</font>
<font style="color:rgb(54, 54, 54);">Begin by downloading the training corpora, which contain verses from the Quran and the Bible (split into Old and New Testaments),</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">here</font>](https://uoe-my.sharepoint.com/:u:/g/personal/bross3_ed_ac_uk/EV3gCLNRxvZJpvbWvviaeM0BHFCkkL981hBaFdEwi8C7Gg?e=6yBOal)<font style="color:rgb(54, 54, 54);">. For this coursework, we will consider the Quran, New Testament, and Old Testament each to be a separate corpus, and their verses as individual documents.</font>

<font style="color:rgb(54, 54, 54);">NOTE: The datasets are slightly different to the ones used for lab 1, so please make sure you use the ones linked to here.</font>

<font style="color:rgb(54, 54, 54);">Each line in the file contains the corpus_name and the text which contains the content of a single verse from the corpus, with the two fields separated by a single tab. Complete the following analyses:</font>

+ <font style="color:rgb(54, 54, 54);">Preprocess the data as usual, including tokenization, stopword removal, etc. Note: you may reuse code from previous labs and courseworks to achieve this.</font>
+ <font style="color:rgb(54, 54, 54);">Compute the Mutual Information and χ</font><sup><font style="color:rgb(54, 54, 54);">2</font></sup><font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">scores for all tokens (after preprocessing) for each of the three corpora. Generate a ranked list of the results, in the format token,score.</font>
+ <font style="color:rgb(54, 54, 54);">In your report, add a section called "Token Analysis" to discuss the following:   
</font><font style="color:rgb(54, 54, 54);">- What differences do you observe between the rankings produced by the two methods (MI and χ</font><sup><font style="color:rgb(54, 54, 54);">2</font></sup><font style="color:rgb(54, 54, 54);">)?   
</font><font style="color:rgb(54, 54, 54);">- What can you learn about the three corpora from these rankings?   
</font><font style="color:rgb(54, 54, 54);">- Include a table in your report showing the top 10 highest scoring words for each method for each corpus.</font>
+ <font style="color:rgb(54, 54, 54);">Run LDA on the entire set of verses from ALL corpora together. Set k=20 topics and inspect the results.   
</font><font style="color:rgb(54, 54, 54);">- For each corpus, compute the average score for each topic by summing the document-topic probability for each document in that corpus and dividing by the total number of documents in the corpus.   
</font><font style="color:rgb(54, 54, 54);">- Then, for each corpus, you should identify the topic that has the highest average score (3 topics in total). For each of those three topics, find the top 10 tokens with highest probability of belonging to that topic.</font>
+ <font style="color:rgb(54, 54, 54);">Add another section "Topic Analysis" to your report which includes:   
</font><font style="color:rgb(54, 54, 54);">- A table including the top 10 tokens and their probability scores for each of the 3 topics that you identified as being most associated with each corpus.   
</font><font style="color:rgb(54, 54, 54);">- Your own labels for the 3 topics. That is, in 1-3 words, what title would you give to each of the three topics?   
</font><font style="color:rgb(54, 54, 54);">- What does the LDA model tell you about the corpus? Consider the three topics you have presented as well as the other topics and their scores for each corpus. Are there any topics that appear to be common in 2 corpora but not the other? What are they and what are some examples of high probability words from these topics? How is this different from the things you learned when analysing the data using MI and χ</font><sup><font style="color:rgb(54, 54, 54);">2</font></sup><font style="color:rgb(54, 54, 54);">?</font>

## <font style="color:rgb(54, 54, 54);">3. Text Classification</font>
<font style="color:rgb(54, 54, 54);">In this part, you are required to implement a sentiment analyser. Sentiment analysis is a task that aims to classify the text based on its polarity. For this coursework, we are considering three-way sentiment classification, where the classes are: positive, negative, or neutral. You can access the training data from</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">here</font>](https://uoe-my.sharepoint.com/:t:/g/personal/bross3_ed_ac_uk/ESSQm4ZpO95Fkx8s0psyxtwBpUu4v5ZOGgQi3_QudyskXw)<font style="color:rgb(54, 54, 54);">. The test data can be downloaded</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">here</font>](https://uoe-my.sharepoint.com/:t:/g/personal/bross3_ed_ac_uk/EWOiv0fTHBNKpC8Y_r_mlccBRVXWJ2DeFSWi5sW2XH2SIg?e=B7GchD)<font style="color:rgb(54, 54, 54);">, it was made available on 24 November (four days before the deadline). The task is: given a text (tweet), predict its sentiment (positive, negative, or neutral). Complete the following steps:</font>

+ <font style="color:rgb(54, 54, 54);">Shuffle the order of the data and split the dataset into a training set and a separate development set. You can split the data however you like. For example, you can use 90% of the documents for training and 10% for testing.</font>
+ <font style="color:rgb(54, 54, 54);">Apply the steps in the text classification lab (lab 7) to this new dataset in order to get your baseline model: extract BOW features and train an SVM classifier with c=1000 to predict the labels (i.e., the sentiment). Note that the input data format is slightly different this time, but you will still need to convert to BOW features. You may reuse your code from the lab.</font>
+ <font style="color:rgb(54, 54, 54);">Compute the precision, recall, and f1-score for each of the 3 classes, as well as the macro-averaged precision, recall, and f1-score across all three classes. Only train the system on your training split, but evaluate it (compute these metrics) on both the training and development splits that you created (i.e., don't train on tweets from the development set).</font>
+ <font style="color:rgb(54, 54, 54);">Identify 3 instances from the development set that the baseline system labels incorrectly. In your report, start a new section called "Classification" and provide these 3 examples and your hypotheses about why these were classified incorrectly.</font>
+ <font style="color:rgb(54, 54, 54);">Based on those 3 examples and any others you want to inspect from the development set, try to improve the results of your classifier (you should have already experimented with ways to do this in the lab). You can make any changes you like. For example, for the SVM, you may change the preprocessing, feature selection (e.g., only using the top N features with the highest MI scores), SVM parameters, etc. You can try different classifiers, whether it is one that you train from scratch or an existing one that you either evaluate as-is or fine-tune further. You can try including word embeddings as features. You can try using LLMs with zero-shot classification. Your goal is to create a system that will perform well on the test set. Until the test set becomes available, you can use performance on the development set as an indication of how well you are doing.</font>
+ <font style="color:rgb(54, 54, 54);">A few marks are given for managing to improve over the baseline, but the bulk of marks are</font><font style="color:rgb(54, 54, 54);"> </font>_<font style="color:rgb(54, 54, 54);">not</font>_<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">based on the final performance of your classifier, they are based on your description of what you tried and why (see below).</font>
+ **<font style="color:rgb(54, 54, 54);">The test set can be downloaded</font>****<font style="color:rgb(54, 54, 54);"> </font>**[**<font style="color:rgb(0, 102, 204);">here</font>**](https://uoe-my.sharepoint.com/:t:/g/personal/bross3_ed_ac_uk/EWOiv0fTHBNKpC8Y_r_mlccBRVXWJ2DeFSWi5sW2XH2SIg?e=B7GchD)**<font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">(as of 24 November, four days before the deadline). Without making any further changes to your baseline or improved models, train on your training set and evaluate on the new test set. Report all of your results in a file called classification.csv with the following format:</font>

```plain
system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro 
    baseline,train,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    baseline,dev,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    baseline,test,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    improved,train,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    improved,dev,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    improved,test,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

<font style="color:rgb(54, 54, 54);">where the 0 values are replaced with the scores you computed for each class or the macro-average, according to the column names from the header. Each row represents the performance for one of the models (baseline or improved) for one of the splits in the dataset (train, dev, or test).   
</font><font style="color:rgb(54, 54, 54);">in the header, p=precision, r=recall, f=f1-score, pos=positive, neg=negative, neu=neutral, macro=macro-averaged scores across all 3 corpora. That is, "p-pos" means the precision score for the positve class, "r-pos" is the recall for that class, and so on.   
</font><font style="color:rgb(54, 54, 54);">- check the format of your file using this</font><font style="color:rgb(54, 54, 54);"> </font>[<font style="color:rgb(0, 102, 204);">script</font>](https://uoe-my.sharepoint.com/:u:/g/personal/bross3_ed_ac_uk/EXIHb2rm7u5ErZx4VIbIJFcBsqtUcYYNgJkRq5SrFAR8WA?e=BV0qug)<font style="color:rgb(54, 54, 54);">.</font>

+ <font style="color:rgb(54, 54, 54);">In your report on this assignment, in the "Classification" section, please explain how you managed to improve the performance compared to the baseline system, and mention how much gain in the Macro-F1 score you could achieve with your improved method when evaluted on the dev set, and how much gain on the test set. Why did you make the changes that you did?   
</font><font style="color:rgb(54, 54, 54);">- Note: it is okay if your test results are different from the development set results, but if the difference is significant, please discuss why you think that is the case in your report.</font>

## <font style="color:rgb(54, 54, 54);">Submissions and Formats</font>
<font style="color:rgb(54, 54, 54);">You need to submit the following:</font>

1. **<font style="color:rgb(54, 54, 54);">ir_eval.csv</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">containing the IR evaluation scores in the format described above.</font>
2. **<font style="color:rgb(54, 54, 54);">classification.csv</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">containing the classification results of the baseline system and the improved system on the train, dev, and test splits as described above.</font>
3. **<font style="color:rgb(54, 54, 54);">code.py</font>**<font style="color:rgb(54, 54, 54);">: a</font><font style="color:rgb(54, 54, 54);"> </font><u><font style="color:rgb(54, 54, 54);">single file</font></u><font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">including all the code that produces the results that you are submitting and discussing in your report   
</font><font style="color:rgb(54, 54, 54);">- If you will use something other than Python, let us know before submission!   
</font><font style="color:rgb(54, 54, 54);">- Please try to make your code as readable as possible: commented code is highly recommended.</font>
4. **<font style="color:rgb(54, 54, 54);">Report.pdf</font>**<font style="color:rgb(54, 54, 54);">: Your report on the work, no more than 6 pages. It should contain:   
</font><font style="color:rgb(54, 54, 54);">- 1 page on the work you did in the assignment in general, which can include information on your implementation code, summary on what was learnt, challenges faced, comment on any missing part in the assignment.   
</font><font style="color:rgb(54, 54, 54);">- 1 page on the best performing IR system for each score (you can put in a table), and an explanation of if the best system is significantly better than the second system or not, and why.   
</font><font style="color:rgb(54, 54, 54);">- 1-2 pages describing the text analysis including MI, χ</font><sup><font style="color:rgb(54, 54, 54);">2</font></sup><font style="color:rgb(54, 54, 54);">, and LDA results and discussion about them.   
</font><font style="color:rgb(54, 54, 54);">- 1-2 pages on the work you did on classification. Describe what you did and why: you should write about how you managed to improve over the baseline, and how it was achieved (new features? different classifiers? more training data? ... etc.). This section can, for example, include information on where you got the idea, what did not work as expected, what you tried next, .. Briefly comment on the final results (performance on dev set vs performance on test set).</font>

<font style="color:rgb(54, 54, 54);">Submit the files listed above on Learn.</font>

## <font style="color:rgb(54, 54, 54);">Marking</font>
<font style="color:rgb(54, 54, 54);">The assignment is worth</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">20%</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">of your total course mark and will be scored out of</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">100 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">as follows:</font>

+ **<font style="color:rgb(54, 54, 54);">20 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">for the output of the IR Evaluation: ir_eval.csv. These marks will be assigned automatically, so you</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">must</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">follow the format (remember that you can check that with the provided script).</font>
+ **<font style="color:rgb(54, 54, 54);">10 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">for the explanation in the report to the best IR system for each score and if it is significant or not.</font>
+ **<font style="color:rgb(54, 54, 54);">35 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">for your presentation of text analysis results in the report, and discussion about what you learned from them.</font>
+ **<font style="color:rgb(54, 54, 54);">10 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">for submitting classification results in the correct format with evidence of improvements above the baseline.</font>
+ **<font style="color:rgb(54, 54, 54);">25 points</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">for discussion about error cases, what you did to improve on the baseline, and your analysis of the final results.</font>

## <font style="color:rgb(54, 54, 54);">Allowed / NOT Allowed</font>
+ <font style="color:rgb(54, 54, 54);">For the IR measures, scores should be 100% calculated with your own code. It is</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">NOT</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">allowed to use ready implementations of these scores. Only for the ttest, you can use libraries (or any tool) to do it.</font>
+ <font style="color:rgb(54, 54, 54);">For the text analysis, you can use code from the comparing corpora lab. You should</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">NOT</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">use existing implementations of mutual information or Χ2, but you</font><font style="color:rgb(54, 54, 54);"> </font>**<font style="color:rgb(54, 54, 54);">are permitted</font>**<font style="color:rgb(54, 54, 54);"> </font><font style="color:rgb(54, 54, 54);">to use any existing implementation or tool for the LDA-based topic modelling.</font>
+ <font style="color:rgb(54, 54, 54);">For the classification, you can directly use your work in the text classification lab. However, your mark depends on the amount of work you put in (e.g. trying different preprocessing steps, defining your own features, trying different models...), so just changing a parameter or two compared with the lab is unlikely to result in a high mark.</font>

