# Multi-label-Senteces-tagging

## Competetion
This is the repo about the competetion hold at T-brain AI CUP platform. [Here is the link](https://tbrain.trendmicro.com.tw/Competitions/Details/8) for the competetion.  
We, as a team, each member had his/her own models. My model **did not** use BERT or ELMO but still get a quite good result without being too complicated.   
If you have resources for BERT or ELMO, please see [our team leader's repo](https://github.com/eugeneALU/Text-Classification).

## Rule
### Target
The target is to predict the labels of each sentence in abstract of papers. These papers come from different domains published in arXiv. And each sentence might have one or more labels, which is a multi-label classification problem.
### Evaluation Metric
Micro F1 score

## Data description
Please follow the feature description provided on competetion.  
And the target labels contain: Background, Objectives, Methods, Results, Conclusions and Others, totally 6 categories.

## Features / tools (simplified steps)
**Basic trial** will give a baseline result (at least 0.67XX or above in public leaderboard).   
**More trial** would only give a subtile progress (0.68XX or above)
### Basic trial
+ Text mining skills (details are described below)
+ GloVe word embedding
+ The raw sentences (current sentence) with word embedding
+ The position of appearence of each label in a sentence (current sentence)
+ Bidirectional GRU implemented with Keras
### More trials
+ Basic features plus the features of previous/following sentences
+ Cosine similarity between each sentence and the title

## Overall steps (detailed steps)
### Preprocessing
There are 3 steps in ```preprocessing.py```. You can see the details in code comments.   
One thing our team found is that **do not** remove stop words and **do not** do stemming can get a better result in this competetion.  
After runnung ```preprocessing.py```, you will get ```after_preprocess_train.csv``` and ```after_preprocess_test.csv``` for training and testing afterwards. 

### Training


### Testing
