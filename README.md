# Sentiment-Analysis
A simple sentiment analysis as an assignment

I have done 3 sentiment analysis for the app reviews of Kuku FM on google play store. 
1) A simple rules based approach using VADER
2) Using Logistic regression and tf-idf
3) A Pre-trained roberta model trained using twitter data

Logistic Regression works the best. And BERT model came last.

This can be explained as the BERT model was not fine tuned on the dataset but just evaluated on test set. Also for short form text like App Reviews traditional ML methods are much better than Deep Learning methods. They are cheaper to train and easier to run. 

This could mainly be because of the limited context available in the review format with a hardly a paragraph at the maximum. As the text sample size increases Deep Learning methods will become more suitable


