import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from textblob import TextBlob
import torch
import spacy
from sklearn.metrics.pairwise import cosine_similarity as cosim
en_nlp = spacy.load('en_core_web_sm')

#loading the data
train = pd.read_json("dev-v2.0.json")

# pre-processing 
contexts = []
questions = []
answers_text = []
answers_start = []
is_impossible = []
for i in range(train.shape[0]):
    topic = train.data[i]['paragraphs']
#    topic = train.iloc[i,0]['paragraphs']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            questions.append(q_a['question'])
            try:
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
            except IndexError:
                answers_start.append("NA")
                answers_text.append("NA")
            
            contexts.append(sub_para['context'])   
            is_impossible.append(q_a['is_impossible'])

# Building a structured Dataframe
df = pd.DataFrame({"context":contexts, "question": questions,"is_impossible":is_impossible, 
                  "answer_start": answers_start, "text": answers_text})

# encoding `is_impossible` into binary
df.is_impossible = df.is_impossible.map(lambda x:0 if x == False else 1)

# creating a list of paragraphs
paras = list(df["context"].drop_duplicates().reset_index(drop= True))

# Sentence-tokenization
blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]

# loading sentence embeddings 
infersent = torch.load('./InferSent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
infersent.set_glove_path("./InferSent/dataset/GloVe/glove.840B.300d.txt")
infersent.build_vocab(sentences, tokenize=True)

# Create a dictionary of each sentence in the paragraph and its embeddings
dict_embeddings = {}
for i in range(len(sentences)):
    dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)


# Create a dictionary of each question and its embeddings
questions = list(df["question"])
for i in range(len(questions)):
    dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)

def get_target(x):
     #input: pandas dataframe containing tokenized sentences and correct answers
	#returns: the index of sentence containing right answer or -1 if the question is unanswerable
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx
	

def cosine_sim(x):
    li = []
    for item in x["sent_emb"][0]:
        li.append(cosim(item,x["quest_emb"][0][0]))
    return li 
def fetch_dep(string):
    # returns a word, dependency tag pair for each sentence
    lst = []
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(string)
    for sent in doc.sents:
        for token in sent:
            lst.append((token.text,token.dep_))
    return lst

def fetch_ner(string):
    #returns a word, ner tag pair for each sentence.
    lst = []
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(string)
    for ent in doc.ents:
        lst.append((ent.text, ent.label_))
    return lst

def jaccard(q,s):
    # returns jaccard score between two lists (texts or tags)
    if type(q) == list:
        a = set(q)
    else:
        a = set(str(q).split(" "))
    if type(s) == list:
        b = set(s)
    else:
        b = set(str(s).split(" "))
    
    return len(a.intersection(b)) / len(a.union(b))

def jac_index(x):
    li = []
    for item in x["sentences"]:
        li.append(jaccard(item,x["question"]))
    return li  

def process_data(train):
    '''
	input: pandas dataframe 
	returns: pandas dataframe with more columns (information)
	'''
    print("step 1") #append word_tokenized sequences to the dataframe
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    
    print("step 2") # gold labels
    train["target"] = train.apply(get_target, axis = 1)
    
    print("step 3") # append sentence embeddings for each sentence in the paragraph
    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_embeddings[item][0] if item in\
                                                           dict_embeddings else np.zeros(4096) for item in x])
    print("step 4") # append sentence embeddings for the question
    train['quest_emb'] = train['question'].apply(lambda x: dict_embeddings[x] if x in dict_embeddings else np.zeros(4096) )
    print("step 5") # append a list of jaccard indices of each question-sentence pair
    train["word_overlap"] = train.apply(jac_index, axis = 1)
    print("step 6") # append a list of cosine of each question-sentence pair
    train.quest_emb.apply(lambda x: x.reshape((4096,)))
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)

    print("step 7") # append the index of sentence with highest cosine sim
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: np.argmax(x))
    print("step 8") # append the index of sentence with highest jaccard
    train["pred_idx_wrdovlp"] = train["word_overlap"].apply(lambda x: np.argmax(x))
        
    return train  


new_df = process_data(df)
