### Question Answering Using SQuAD Dataset

This repo contains an implementation of simple logisitic regression, which is the proposed baseline for the first version of the competition, but with one significant difference. The new version of task SQuAD 2.0 requires models not only to answer the questions, but also to abstain from answering if an answer does not exist. Thus, at this level, the logisitic regression classifier implemented extracts the sentence containing the right answer or abstains from answering. The exact answer span extraction will not be tackled in the phase. 

### Instructions

* Please check Dockerfile
* Non-docker installation: In case docker did not work:
  * `git clone https://github.com/raghebalghezi/squad-final.git`
  * `cd  squad-final`
  * `unzip with_pos_overlap_score.csv.zip`
  * `pip install scikit-learn pandas`
  * `python lr.py`

To simplifying the process, and shorten the runtime, I have done the pre-processing in the background, and converted the dataset from `json` format to`csv` . So, `with_pos_overlap_score.csv` contains the following information:
* **answers** : correct answer span
* **context**: supporting paragraph
* **is_impossible**: whether or not the question is answerable
* **plausible_answers**: if unanswerable, what would be the answer from the paragraph? NOT REQUIRED by the task
* **question**: the question prompt
* **sentences**: the sentence-tokenized version of context
* **target**: the index of sentence containing the correct answer span; -1 if question is unanswerable.
* **cosine_sim**: the Cosine similaity score between the question and each of sentences
* **word_overlap**: Jaccard score of the question and each of sentences
* **pred_idx_cos**: the Index of the sentence containing the answer as predicted by `cosine_sim`; i.e. argmax(cosine_sim)
* **pred_idx_wrdovlp**: the Index of the sentence containing the answer as predicted by `word_overlap`; i.e. argmax(word_overlap)
* **pos_tag_sent**: part-of-speech sequence of each sentence
* **pos_tag_quest**: part-of-speech sequence of question
* **pos_tag_ovrlap**:  Jaccard score of the pos_tag_quest and each of pos_tag_sent

### Important Note

For computational reasons, I used a small partation of the data (only 38K data points out of 130K). If you want to see the classification using the whole data set, kindly modify line #33:

```python
# you can remove the whole thing between the brackets
small_partion = train2.iloc[2000:40000] 
```

There may appear some warnings in the code; they are not affecting the code running.

### Results

​             precision    recall  f1-score   support

​       -1.0       1.00      0.99      1.00      2198

​        0.0       0.39      1.00      0.56      2124

​        1.0       0.00      0.00      0.00      1231

​        2.0       0.00      0.00      0.00       808

​        3.0       0.00      0.00      0.00       540

​        4.0       0.00      0.00      0.00       319

​        5.0       0.00      0.00      0.00       169

​        6.0       0.00      0.00      0.00        92

​        7.0       0.00      0.00      0.00        57

​        8.0       0.00      0.00      0.00        23

​        9.0       0.00      0.00      0.00        20

​       10.0       0.00      0.00      0.00        11

​       11.0       0.00      0.00      0.00         4

​       12.0       0.00      0.00      0.00         1

​       14.0       0.00      0.00      0.00         1

​       15.0       0.00      0.00      0.00         1

​       19.0       0.00      0.00      0.00         1

avg / total       0.40      0.57      0.45      7600      
