import csv
import spacy
import re
import nltk
import pandas as pd

import constants

from typing import Set, List, Dict, Tuple

from datasets import load_dataset, Dataset

NLP = spacy.load("en_core_web_sm")


def word_shape(word: str):
    shape = re.sub("[A-Z]", "X", word)
    shape = re.sub("[a-z]", "x", shape)
    shape = re.sub("[0-9]", "d", shape)
    shape = re.sub(r"\W", "w", shape)
    return shape


def gather_test_bio_ner_tags(file_name: str) -> Set[str]:
    bio_ner_tags = set()
    with open(file_name, "r") as f:
        data = csv.DictReader(f=f, delimiter="\t")
        for row in data:
            bio_ner_tags.add(row["bio_ner_tag"])
            
    return bio_ner_tags


def nerc_data_to_file(raw_data: Dataset, file_name: str):
    try:
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer: csv.writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sentence_id", "token_id", "token", "bio_ner_tag"])
            
            for idx, sent in enumerate(raw_data):
                sentence = " ".join(sent["tokens"])
                doc = NLP(sentence)
                
                bio_tags = ["O"] * len(doc)
                
                for ent in doc.ents:
                    bio_tags[ent.start] = f"B-{ent.label_}"
                    for i in range(ent.start + 1, ent.end):
                        bio_tags[i] = f"I-{ent.label_}"
                        
                for token_id, token in enumerate(doc):
                    writer.writerow([idx, token_id, token.text, bio_tags[token_id]])
                
            f.close()
            
        print("Converted successfully!")
    except Exception as e:
        return {"error": str(e)}


def gather_tokens_and_tags(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    X, y = [], []
    
    sent_tokens = []
    sent_tags = []

    for token, tag in zip(df["token"], df["bio_ner_tag"]):
        sent_tokens.append(token)
        sent_tags.append(tag)
        
        if token in [".", "!", "?"]:
            X.append(sent_tokens)
            y.append(sent_tags)
            sent_tokens = []
            sent_tags = []
            
    if sent_tokens:
        X.append(sent_tokens)
        y.append(sent_tags)
        
    return X, y
    
    
    
def sentiment_data_to_file(raw_data: Dataset, file_name: str):
    try:
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer: csv.writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sentence_id", "sentence", "sentiment"])
            
            for idx, elem in enumerate(raw_data):
                sentence = elem["sentence"]
                label = "positive" if elem["label"] == 1 else "negative"
                
                writer.writerow([idx, sentence, label])
            
        f.close()
        print("Converted successfully!")
    except Exception as e:
        return {"message": str(e)}
    

def topic_data_to_file(raw_data: Dataset, file_path: str):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter = "\t")
        writer.writerow(["id", "question", "category"])
        for entry in raw_data:
            id_ = entry["id"]
            q = entry["question"]
            category = entry["category"]
            
            if category == "movies":
                category = "movie"
            elif category == "books":
                category = "book"
            
            writer.writerow([id_, q, category])
            
        f.close()
        
        
def extract_features(sentence, pos_tags, i):
    word = sentence[i]
    pos = pos_tags[i]
    
    if not isinstance(word, str):
        word = str(word)
    
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "pos": pos,
        "word.shape": word_shape(word=word)
        
    }
    if i > 0:
        word1 = sentence[i-1]
        pos1 = pos_tags[i-1]
        
        if not isinstance(word1, str):
            word1 = str(word1)
        
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:pos": pos1,
            "-1:word.shape": word_shape(word=word1)
        })
    else:
        features["BOS"] = True

    if i < len(sentence) - 1:
        word1 = sentence[i+1]
        pos1 = pos_tags[i+1]
        
        if not isinstance(word1, str):
            word1 = str(word1)
            
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
            "+1:pos": pos1,
            "+1:pos": word_shape(word=word1)
        })
    else:
        features["EOS"] = True

    return features

def sentence_to_features(sentence):
    cleaned_sentence = [str(token) if not isinstance(token, str) else token for token in sentence]
    pos_tags = [pos for _, pos in nltk.pos_tag(cleaned_sentence)]
    return [extract_features(cleaned_sentence, pos_tags, i) for i in range(len(sentence))]
        
