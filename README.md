# Stock Prediction Website (Vision) 

**Vision is a stock prediction website that aims to utilize sentiment analysis and deep learning models to predict future stock prices.**


This project is still under development.  


## Tools:
The website is mainly implemented with Python. 



## Long short-term memory (LSTM) 
The LSTM model will use closing prices of stocks of the last 60 days as input and output the predicted stock price of the next day. The current model uses the closing price of AAPL from 2012-01-03 to 2020-01-31 as training data. (Yahoo Finance) 


## Naive Bayes Model (NLP) 
The Naive Bayes model will classfy whether the input sentence is positive or negative and output the confidence level. This model utilized several classifiers and used the summarized result as prediction. The model uses classified Financial news as training data. 


## Sentiment analysis 
Details: @ https://github.com/michaelchenghw/sentiment-analysis


## Website 
The website will be developed in Flask. 



 

## References and Documentation 
1. https://www.datacamp.com/community/tutorials/lstm-python-stock-market
2. https://keras.io/api/layers/recurrent_layers/lstm/
3. http://www.cs.cornell.edu/courses/cs5740/2017sp/lectures/11-rnn.pdf
4. http://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf
5. https://medium.com/@eiki1212/natural-language-processing-naive-bayes-classification-in-python-e934365cf40c
6. https://towardsdatascience.com/algorithms-for-text-classification-part-1-naive-bayes-3ff1d116fdd8

