# Text based Emotion Recognition across BERT Family and LSTM



### Description of project description sections

1. __Project goals__: Overall, the current paper's objective is to develop a model that will predict an emotion from a given string of text. Our main goal is to replicate and generalize the study on data sets of similar structure, but with different emotional categories to further assess the BERT, RoBERTa, DistilBERT, and XLNet model validity and performance. By evaluating various models on separate datasets, we can also gain a sense of their efficacy and computational requirements in different use cases.
2. __NLP task description__: This project falls into the domains of both text classification and sentiment analysis. We are given a sentence coming from a tweet and the objective is to assign a categorical label based on its tone and emotion.
3. __Project data__: 
We plan to use a dataset similar to the one discussed in the paper. It consists of short twitter messages (1 sentence) followed by different labels categorizing their sentiment. These labels are organized the following way:

0 - sadness  
1 - joy  
2 - love  
3 - anger  
4 - fear  
5 - surprise  

Link: https://huggingface.co/datasets/dair-ai/emotion  
4. __Neural methodology__: ??  
5. __Baselines__: We'll use a previous generation state-of-the-art model, LSTM as our baseline.  
6. __Evaluation__: We will compute the R, F1 and P score of each model for each emotion category because these metrics are the golden standards for classification task. In the meanwhile, we may also compare models' trainig time to rank their efficiancy.  



## Paper summary metadata
- Title: Comparative Analyses OF BERT, ROBERTA, DISTILBERT, andd XLNET For Text-Based Emotion Recognition
- Link: https://ieeexplore.ieee.org/document/9317379
- Bibliographical information: Achempong Francisca Adoma,
, Nunoo-Mensah Henry
, Wenyu Chen

**Publisher:** Institute of Electrical and Electronics Engineers (IEEE)   
**Published in:** 2020 17th International Computer Conference on Wavelet Active
Media Technology and Information Processing (ICCWAMTIP)


* The paper discusses the significance of text-based emotion recognition in the context of the growing number of social media users and the need for more fine-grained methods of user profiling. It highlights the limitations of existing techniques and introduces the concept of transformers, specifically the BERT model, as a breakthrough in addressing these limitations. The paper also introduces other transformer models, including RoBERTa, DistilBERT, and XLNet, which were developed to mitigate certain issues associated with BERT. It positions the comparative analysis of these models as a key contribution to the field of NLP research.

* 
