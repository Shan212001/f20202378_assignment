from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import json
import re
from typing import List, Dict
import random
import joblib 

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

def load_domain_knowledge(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        return json.load(file)

domain_dict = load_domain_knowledge("final_extracted_entities_output.json")

model = joblib.load("final_trained_model.pkl")  

def predict_labels(text: str) -> List[str]:
    prediction = model.predict([text])
    
    predicted_labels = [str(label) for label, value in enumerate(prediction[0]) if value == 1]
    
    return predicted_labels

def dictionary_lookup(text: str, domain_dict: Dict) -> List[str]:
    found_keywords = []
    for category, keywords in domain_dict.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
                found_keywords.append(keyword)
    return found_keywords

#Named Entity Recognition (NER)
def extract_entities_ner(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def generate_summary(text: str) -> str:
    sentences = text.split(".")
    return " ".join(sentences[:2]) + "."

#request model for the incoming JSON data
class TextSnippet(BaseModel):
    text: str

@app.post("/process")
async def process_text(snippet: TextSnippet):
    text = snippet.text

    labels = predict_labels(text)

    dictionary_entities = dictionary_lookup(text, domain_dict)
    ner_entities = extract_entities_ner(text)
    combined_entities = list(set(dictionary_entities + ner_entities))

    summary = generate_summary(text)

    return {
        "Predicted Labels": labels,
        "Extracted Entities": combined_entities,
        "Summary": summary
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
