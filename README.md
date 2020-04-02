# Multi-label-Senteces-tagging
## Competetion
This is the repo about the competetion hold at T-brain AI CUP platform. [Here is the link](https://tbrain.trendmicro.com.tw/Competitions/Details/8) for the competetion.
We, as a team, each member had his/her own models. My model did not use BERT or ELMO but still get a quite good result without being too complicated (0.68 in public leaderboard, but not selected to be the final submittion so not sure the final public score). If you have resources for BERT or ELMO, please see [our team leader's repo](https://github.com/eugeneALU/Text-Classification)
## Rule
The target is to predict the labels of each sentence in abstract of papers. These papers come from different domains published in arXiv. And each sentence might have one or more labels, which is a multi-label classification problem.
## Data description
Please follow the feature description provided on competetion.
And the target labels contain: Background, Objectives, Methods, Results, Conclusions and Others, totally 6 categories.
## Features
+ Ofcourse the raw sentences
+ The position of appearence of each label in a sentence (Might be an important feature)
