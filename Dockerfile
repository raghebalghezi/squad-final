FROM python:3

CMD wget https://github.com/raghebalghezi/squad-final/blob/master/with_pos_overlap_score.csv.zip; unzip with_pos_overlap_score.csv.zip

RUN pip install scikit-learn pandas
