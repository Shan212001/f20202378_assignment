# DATASET

## Input Data (CSV):
- The CSV file contains textual descriptions (e.g., product information, competitor strategies, and technology trends) relevant to the EV charging domain.
- Each row includes: Description: A snippet of text providing details about EV charging technologies, pricing strategies, and regional presence.

## Domain Knowledge Base (JSON): The JSON file provides pre-defined categories, entities, and keywords to enhance entity extraction and multi-label classification.
- Categories include:
1. Competitors: EV charging-related companies and organizations (e.g., Tesla Supercharger, ChargePoint).
2. Features: Key attributes of EV charging infrastructure (e.g., charging speed, battery swapping).
3. Pricing Keywords: Keywords associated with pricing strategies and cost optimization (e.g., cost-effective charging, competitive pricing).
4. Regions: Geographical regions of interest (e.g., North America, Europe).
5. Sectors: Industries impacted by or involved in EV charging (e.g., Technology, Energy).

# SAMPLE OF THE TASK

## input 
```bash
Description: "Tesla Supercharger leads the EV charging network with unparalleled charging speed and a vast network across North America."
```

## input 
```bash
{
  "competitors": ["Tesla Supercharger"],
  "features": ["charging speed", "charging network coverage"],
  "regions": ["North America"]
}
```

## input 
```bash
{
  "Predicted Labels": ["Competitors", "Features", "Regions"],
  "Extracted Entities": ["Tesla Supercharger", "charging speed", "North America"],
  "Summary": "Tesla Supercharger leads the EV charging network with unparalleled charging speed."
}
```



# Task 1: Data Preparation & Multi-Label Text Classification

## Overview
In this task, we focus on the **data preparation** and **multi-label text classification** for our model. The main objective is to train and validate a multi-label classification model using text data. The process includes data preprocessing, model training, and evaluation. We have organized the code for this task into a Jupyter notebook called `Pretrain_validate.ipynb`.

## Key Steps

### 1. **Data Preparation**
The model is pre-trained using the data from the `input.csv` file. This file contains the raw data which will be processed, cleaned, and formatted for the training pipeline. The CSV file is structured in a way that each row corresponds to a text instance, and each instance is associated with one or more labels for multi-label classification. Alsodata augmentation for minority label was done.

### 2. **Preprocessing**
The data is preprocessed to remove any irrelevant information, handle missing values, and normalize the text. Text processing steps such as tokenization, lowercasing, and padding are applied to ensure uniformity and consistency before feeding the data into the model.

### 3. **Model Training**
We use the preprocessed text data to pretrain a multi-label classification model. The training involves utilizing suitable techniques for multi-label classification, such as binary cross-entropy loss and one-vs-rest (OvR) strategies for multi-class classification. We also split the dataset into training and validation sets to evaluate the model's performance during training. 5 fold Cross-validation is done for a robust performance estimate.

### 4. **Model Validation**
After training the model, we validate it using a set of metrics to assess its accuracy in predicting multiple labels for each text instance. Metrics such as **accuracy**, **precision**, **recall**, and **F1-score** are calculated for the evaluation.

---

## Files & Resources

- **`Pretrain_validate.ipynb`**: This Jupyter notebook contains the complete code for training, validation, and evaluation of the multi-label classification model. It also includes code for preprocessing the data and training the model.
  
- **`input.csv`**: This CSV file contains the input data with text instances and their corresponding labels. The dataset is used for pretraining the model.

---

## Setup & Requirements

To run the code in `Pretrain_validate.ipynb`, you will need to have the following dependencies installed:

- Python 3.x
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- TensorFlow/Keras (for model training)
- NLTK or spaCy (for text preprocessing)

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn numpy matplotlib tensorflow nltk spacy
```

---

## Running the Code

1. Open the `Pretrain_validate.ipynb` notebook in Jupyter Notebook.

2. Run the notebook cells in order to preprocess the data, train the model, and evaluate it.

3. Ensure that the `input.csv` file is in the same directory as the notebook or update the file path accordingly in the code.

---

## Results

The results of the model training and validation will be displayed at the end of the notebook. The notebook includes evaluation metrics for the multi-label classification task.

---

# Task 2: Entity/Keyword Extraction with a Domain Knowledge Base

## Overview
In this task, we performed entity and keyword extraction from the input text using a pre-defined domain knowledge base. The goal was to identify and extract relevant entities and keywords to enrich our model's understanding of the domain and improve the subsequent classification tasks.

## Steps

### 1. Loading the Domain Knowledge Base:
- The knowledge base was loaded from a `input.json`, `input.csv` file that contained predefined terms, entities, and keywords related to the domain.

### 2. Preprocessing the Input Data:
- The text data was preprocessed to remove any unnecessary characters, such as special symbols or whitespace.
- Tokenization and lowercasing were applied to ensure uniformity in entity extraction.

### 3. Keyword Extraction:
- Methods like **TF-IDF** (Term Frequency-Inverse Document Frequency) or simple **keyword matching** were used to identify important keywords within the text.

### 4. Entity Recognition:
- For entity extraction, techniques like **Named Entity Recognition (NER)** or **dictionary-based matching** were employed, using the domain-specific knowledge base to find entities relevant to the task.

### 5. Post-processing:
- Post-processing steps included filtering out irrelevant terms and ensuring the accuracy of extracted entities and keywords.

### 6. Saving the Results:
- Extracted entities and keywords were saved to a file, typically in a `.json` format.

## Code
All code for this task has been saved in the file `ner_extraction.ipynb`.

## Input
The input to this task are `input.json` & `input.csv` file that contains raw text data. This file is preprocessed and used for entity and keyword extraction.

## Output
The output is a file, typically in `.json` format, containing the extracted entities and keywords from the input text.

---

# Task 3: Containerized Inference Service

## Overview
In this task, we created a REST API that performs three main tasks for a given text snippet:
1. Predicts the labels (multi-label classification).
2. Extracts entities from a domain knowledge base and using Named Entity Recognition (NER).
3. Generates a summary of the input text.

We then containerized the service using Docker.

## Steps Implemented

### Step 1: Implement the REST API with FastAPI

We used **FastAPI** to create the REST API that accepts a JSON representing a text snippet and returns the following:
- **Predicted Labels**: Multi-label classification based on a pre-trained model.
- **Extracted Entities**: Entities identified using both a domain knowledge base (dictionary lookup) and Named Entity Recognition (NER) using spaCy.
- **Summary**: A simple summary of the text (first 1-2 sentences).

### Step 2: Dockerize using Docker

- We used Docker to implement the bucket of the files and its requirements.
- The requirements of the app are saved in the 'app' folder.
  
## Input
```bash
docker build -t my-fastapi-app .
```

- This will use main.py and other requirements to host the pipeline.

```bash
docker run -d -p 8000:8000 my-fastapi-app
```

To run the app on port 8000.

- Open any web browser and run "http://localhost:8000/docs" to run the app.
- Try It Out > In place of "text":"string", mention the string that we want to give as a input and then click on execute.
- Alternatively we can also use Curl method to implement this.
- For eg. 
```bash
curl -X 'POST' \
  'http://localhost:8000/process' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "SIEMENS LAUNCHED A NEW EV"
}
```

## Hosting on heroku

- In order to host the service on Heroku we need a paid subscription, which will be free for 750 dyno hours, not sure of the implementation havent used the hosting platform and host it using FastApi only.


