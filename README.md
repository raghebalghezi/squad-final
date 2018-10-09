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
small_partion = train2.iloc[:] 
```

There may appear some warnings in the code; they are not affecting the code running.

### Results: 

**UPDATE** results reported below are on 30% of the data, but i managed to run it on the whole 13k sentence, and I got much better results. Kindly take a look at them.


                 precision    recall  f1-score   support
         -1       1.00      0.99      0.99      8872
          0       0.47      0.68      0.56      4677
          1       0.63      0.54      0.58      3869
          2       0.62      0.55      0.58      3090
          3       0.62      0.53      0.57      2321
          4       0.58      0.47      0.52      1412
          5       0.57      0.48      0.52       799
          6       0.57      0.49      0.53       429
          7       0.46      0.33      0.38       247
          8       0.56      0.38      0.45       124
          9       0.46      0.37      0.41        89
         10       0.48      0.32      0.39        50
         11       0.35      0.21      0.26        34
         12       0.00      0.00      0.00        11
         13       0.00      0.00      0.00        15
         14       0.00      0.00      0.00        10
         15       0.00      0.00      0.00         4
         16       0.00      0.00      0.00         5
         17       0.00      0.00      0.00         2
         18       0.00      0.00      0.00         1
         20       0.00      0.00      0.00         2
         21       0.00      0.00      0.00         1
                  0.72      0.71      0.71     26064
