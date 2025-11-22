# AG News Text Classification

## Objective
Classify news articles from the AG News Dataset using both traditional feature-based models (TF and TF-IDF with Naive Bayes) and deep learning approaches (CNNs with trainable or pre-trained GloVe embeddings, and fine-tuned DistilBERT). Compare performance across these approaches to assess the impact of model complexity and embedding quality.

## Structure
- `notebooks/traditional_vs_cnn_text_classification.ipynb` - traditional models (TF and TF-IDF with Naive Bayes) and CNN approaches.
- `notebooks/distilbert_text_classification.ipynb` - fine-tuned DistilBERT approach
- `requirements.txt` - Python dependencies

## Methods
- **Traditional / CNN notebook:**
  - Preprocess text data.
  - Train TF and TF-IDF with Naive Bayes; record training and test accuracy.
  - Train CNN with random embeddings and CNN with fixed GloVe embeddings; record training/validation and test accuracy.
- **DistilBERT notebook:**
  - Preprocess and clean text (lowercasing, removing punctuation/special characters).
  - Tokenise text using the DistilBERT tokenizer.
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
- Test accuracy improves consistently with increasing model complexity and using richer embeddings.

## How to Run:
pip install -r requirements.txt
jupyter notebook notebooks/traditional_vs_cnn_text_classification.ipynb
jupyter notebook notebooks/distilbert_text_classification.ipynb

## Summary
This project demonstrates that deep learning architectures outperform traditional feature-based methods for text classification. Leveraging pretrained embeddings (GloVe) and fine-tuned transformer models (DistilBERT) further improves performance, highlighting the benefits of increased model complexity and semantic representation quality.


