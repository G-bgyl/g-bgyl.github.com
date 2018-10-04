---
layout: post
title: SMS type classification with Tensorflow
category: machine learning
tags: [Tensorflow, Neural Network, Machine Learning]
---
This  week's task is to use neural network to classify short messages.

### Word2Vec
For LSTM, I need to translate words into vectors for network to learn.I chose to use Tensorflow with LSTM cells to tackle this problem. I used [Chinese Word2Vec](https://github.com/Kyubyong/wordvectors) as pretrained Word2Vec vocabulary. Because vectors list for each word seperates in several lines, so we need to 
```
loadw2v_start_time = time.time()
with open (zh_word_vec_file,'r')as word2vec:
    lines = csv.reader(word2vec)
    vocab_str = []
    vocab_vec = []
    i=0
    for line in lines:
        i+=1
        # each line is a list with only one string.
        str_line = line[0]

        # process word vector that is not strictly a list

        if '\t' in str_line: # means this line contains the word segment string
            one_seg = str_line.split('\t')
            str_word_seg = one_seg[1]
            vec_word_seg = one_seg[2]
        elif ']' in str_line: # last line of a word segment's vector
            vec_word_seg += str_line
            vocab_str.append(str_word_seg)
            # turn string into list and then convert list of strings into list of float
            vec_word_seg = list(map(float, vec_word_seg.strip('[]').split()))
            #convert list into np array
            vocab_vec.append(np.asarray(vec_word_seg))
            str_word_seg=''
            vec_word_seg=[]
        else:# not finish a word segment's vector
                    vec_word_seg += str_line
```

### s
### References
[Chinese Word2Vec](https://github.com/Kyubyong/wordvectors)
[Jieba](https://github.com/fxsjy/jieba)
[Perform sentiment analysis with LSTMs, using TensorFlow](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow)
[USING LOGISTIC REGRESSION IN PYTHON](http://www.dummies.com/programming/big-data/data-science/using-logistic-regression-in-python-for-data-science/)
[如何为Tensorflow训练打包数据和预处理？](https://blog.csdn.net/column/details/16035.html)
[Machine Learning Basics — Part 4 — Anomaly Detection, Recommender Systems and Scaling](https://towardsdatascience.com/machine-learning-basics-part-4-anomaly-detection-recommender-systems-and-scaling-b8bbf0413aa9)

[Introduction to Anomaly Detection](https://www.datascience.com/blog/python-anomaly-detection)
