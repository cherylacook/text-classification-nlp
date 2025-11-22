# AG News Text Classification

## Objective
Classify news articles from the AG News Dataset using both traditional feature-based models (TF and TF-IDF with Naive Bayes) and deep learning approaches (CNNs with trainable or pre-trained GloVe embeddings, and fine-tuned DistilBERT). Compare performance across these approaches to assess the impact of model complexity and embedding quality.

## Structure
- `data/train.csv` - Training data for AG News.
- `data/test.csv` - Test data for AG News.
- `notebooks/traditional_vs_cnn_text_classification.ipynb` - Traditional models and CNN approaches.
- `notebooks/distilbert_text_classification.ipynb` - Fine-tuned DistilBERT.
- `requirements.txt` - Python dependencies.

## Dataset
This project uses the AG News dataset provided in the repository as CSV files:
- `data/train.csv` - Training set.
- `data/test.csv` - Test set.

## Methods
- **Traditional / CNN notebook:**
  - Preprocess text data.
  - Train TF and TF-IDF with Naive Bayes; record training and test accuracy.
  - Train CNNs with random embeddings and fixed GloVe embeddings; record training/validation and test accuracy.
- **DistilBERT notebook:**
  - Preprocess and clean text (lowercasing, removing punctuation/special characters).
  - Tokenise text using DistilBERT tokeniser.
  - Fine-tune pretrained DistilBERT with HuggingFace Trainer and evaluate on the test set.
 
## Key Results
- **Test Accuracy:**
  - *Traditional Models:*
    - TF: 90% 
    - TF-IDF: 90.5%
  - *CNNs:*
    - Random embeddings: 90.9%
    - Fixed GloVe embeddings: 91.3%
  - *Fine-tuned DistilBERT:* 93.4%
- Test accuracy improves consistently with increasing model complexity and leveraging richer embeddings.

## How to Run:
Python version: 3.10+
```bash
pip install -r requirements.txt
jupyter notebook notebooks/traditional_vs_cnn_text_classification.ipynb
jupyter notebook notebooks/distilbert_text_classification.ipynb
```

## Summary
This project demonstrates that deep learning architectures outperform traditional feature-based methods for text classification. Leveraging pretrained embeddings (GloVe) and fine-tuning transformer models (DistilBERT) further improves accuracy, highlighting the value of richer model architectures and semantic representations.


