import csv
import spacy
import pandas as pd

import constants

from typing import Set

from datasets import load_dataset, Dataset

NLP = spacy.load("en_core_web_sm")

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
        
