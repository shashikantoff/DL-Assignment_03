# DL Assignment 03: Sentiment Analysis with RNN and LSTM

This assignment builds a deep learning based sentiment analysis system using the IMDb movie review dataset. The notebook trains two sequence models, a simple Recurrent Neural Network (RNN) and a Long Short-Term Memory (LSTM) network, and compares how well they classify reviews as positive or negative.

📚 Course Details
Course Code: ETMMDL274
Course Name: Deep Learning Architectures and Techniques
Program: MCA (AI/ML)
Semester: 2nd
👨‍🎓 Student Details
Name: Ashish Raj
Roll No: 2501940024

## Objective

The main goal of the notebook is to:

- load a standard text classification dataset,
- preprocess variable-length movie reviews into a fixed-length format,
- train two recurrent deep learning models,
- compare their training behavior, and
- evaluate the final model using classification metrics and a confusion matrix.

In simple terms, the assignment asks: can a neural network learn whether a movie review expresses a positive or negative opinion, and which recurrent model performs better?

## Dataset Used

The notebook uses the IMDb dataset available through `tensorflow.keras.datasets.imdb`.

Key characteristics:

- Total training samples: 25,000
- Total testing samples: 25,000
- Vocabulary size used: 10,000 most frequent words
- Task type: binary classification
- Labels:
  - `0` for negative review
  - `1` for positive review

The reviews are already integer-encoded, which means each review is stored as a sequence of word indices instead of raw text sentences.

## Libraries and Tools

The notebook imports and uses the following main libraries:

- `NumPy` for numerical operations
- `Matplotlib` for plotting loss and accuracy curves
- `TensorFlow / Keras` for loading the dataset and building the models
- `scikit-learn` for the classification report and confusion matrix
- `Seaborn` for visualizing the confusion matrix as a heatmap

## Workflow in the Notebook

The notebook follows a clear deep learning pipeline:

### 1. Load the IMDb dataset

The dataset is loaded with a vocabulary cap of 10,000 words:

- `vocab_size = 10000`

Only the most common words are kept, which helps reduce noise and keeps the model size manageable.

### 2. Pad all review sequences

Movie reviews have different lengths, but neural networks require uniform input shape within a batch. To solve this, the notebook pads all sequences to the same length:

- `max_len = 200`

This means every review is converted into a sequence of 200 tokens. Shorter reviews are padded, and longer ones are truncated.

### 3. Build and train a Simple RNN model

The first model is a basic recurrent neural network with this structure:

- `Embedding(vocab_size, 128)`
- `SimpleRNN(64)`
- `Dense(1, activation='sigmoid')`

What this means:

- The `Embedding` layer converts word indices into dense vector representations.
- The `SimpleRNN` layer processes the review as a sequence.
- The final `Dense` layer outputs a probability for positive sentiment.

Compilation settings:

- Loss function: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

Training settings:

- Epochs: 3
- Batch size: 64
- Validation split: 0.2

### 4. Build and train an LSTM model

The second model replaces the simple RNN with an LSTM layer:

- `Embedding(vocab_size, 128)`
- `LSTM(64)`
- `Dense(1, activation='sigmoid')`

Why LSTM is important:

LSTM networks are designed to handle longer-term dependencies in sequential data more effectively than a basic RNN. In sentiment analysis, this matters because the meaning of a review often depends on word relationships spread across the sentence or paragraph.

Compilation and training settings remain the same:

- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`
- Epochs: 3
- Batch size: 64
- Validation split: 0.2

### 5. Compare training curves

The notebook plots:

- training loss comparison for RNN vs LSTM
- training accuracy comparison for RNN vs LSTM

These graphs help show which model learns more steadily and which one generalizes better during training.

### 6. Evaluate predictions

After training, the notebook uses the LSTM model for final prediction on the test set. It generates:

- a classification report,
- overall accuracy, and
- a confusion matrix heatmap.

## Model Performance Summary

### Simple RNN training behavior

Observed validation performance across 3 epochs:

- Epoch 1: validation accuracy about `0.7890`, validation loss `0.4651`
- Epoch 2: validation accuracy about `0.7904`, validation loss `0.4683`
- Epoch 3: validation accuracy dropped to about `0.6892`, validation loss increased to `0.6899`

Interpretation:

The simple RNN improved on the training set, but its validation performance dropped by the final epoch. This suggests weaker generalization and possible overfitting or instability.

### LSTM training behavior

Observed validation performance across 3 epochs:

- Epoch 1: validation accuracy about `0.8626`, validation loss `0.3311`
- Epoch 2: validation accuracy about `0.8640`, validation loss `0.3188`
- Epoch 3: validation accuracy about `0.8668`, validation loss `0.3354`

Interpretation:

The LSTM performed better than the simple RNN and remained much more stable during training. It achieved stronger validation accuracy and lower validation loss overall.

### Final test-set classification report for LSTM

Saved output from the notebook:

```text
              precision    recall  f1-score   support

           0       0.84      0.88      0.86     12500
           1       0.88      0.84      0.86     12500

    accuracy                           0.86     25000
   macro avg       0.86      0.86      0.86     25000
weighted avg       0.86      0.86      0.86     25000
```

What these numbers mean:

- The model achieves about `86%` test accuracy.
- It performs well on both positive and negative reviews.
- Precision, recall, and F1-score are balanced, which indicates a reliable classifier.

## Confusion Matrix Meaning

The notebook also plots a confusion matrix using Seaborn.

This visualization shows:

- how many negative reviews were correctly predicted as negative,
- how many positive reviews were correctly predicted as positive,
- and where the model made mistakes.

Since the classification report is balanced, the confusion matrix likely shows fairly even performance across both classes.

## Key Conclusion

The assignment demonstrates that:

- recurrent neural networks can be used for sentiment analysis,
- sequence padding is necessary to standardize text input length,
- embedding layers help represent text numerically,
- LSTM handles sequential text data better than a basic SimpleRNN in this task,
- and the LSTM model reaches strong performance on the IMDb dataset with a relatively simple architecture.

In practical terms, the LSTM is the better model in this notebook because it captures review context more effectively and delivers higher and more stable validation and test performance.

## Strengths of the Notebook

- Uses a well-known benchmark dataset
- Compares two different recurrent architectures
- Includes both visual and metric-based evaluation
- Keeps the workflow easy to follow
- Produces a meaningful real-world NLP classification result

## Possible Improvements

If this assignment is extended in the future, the following improvements could make it stronger:

- add markdown explanations inside the notebook for better readability,
- include preprocessing explanation for decoded text examples,
- evaluate both RNN and LSTM on the test set for direct comparison,
- add dropout or regularization to reduce overfitting,
- train for more epochs with early stopping,
- try `GRU`, bidirectional LSTM, or transformer-based models,
- and include ROC-AUC or additional error analysis.

## How to Run the Notebook

To run the work from scratch, make sure the environment has:

- Python 3
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- Seaborn
- Jupyter Notebook or JupyterLab

Typical steps:

1. Open the notebook in Jupyter.
2. Run the cells in order from top to bottom.
3. Allow the IMDb dataset to download if it is not already cached.
4. Wait for both models to train.
5. Review the plots, classification report, and confusion matrix.

## Short Assignment Summary

This assignment is a comparative study of RNN and LSTM models for binary sentiment analysis on IMDb movie reviews. After preprocessing the text sequences and training both models, the notebook shows that the LSTM model clearly outperforms the basic RNN and achieves about `86%` accuracy on the test set. The overall result supports the idea that LSTM is more suitable than a simple RNN for capturing sentiment patterns in sequential text data.
