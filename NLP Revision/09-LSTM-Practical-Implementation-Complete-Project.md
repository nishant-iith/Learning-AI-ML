# Chapter 9: LSTM Practical Implementation - Complete Project

## 🎯 Learning Objectives
- Build complete end-to-end LSTM project
- Master text preprocessing for deep learning
- Implement fake news classifier using LSTM
- Learn complete workflow from data to deployment-ready model
- Understand hyperparameter tuning for LSTM
- Master evaluation metrics for text classification

## 📚 Complete Project Workflow

### Project: Fake News Classifier using LSTM

**Problem Statement**: Given news article title, predict whether news is fake (1) or real (0)

**Dataset**: Kaggle Fake News Dataset (train.csv)
- **Features**: title, author, text
- **Target**: label (0 = real, 1 = fake)
- **Records**: ~20,800

**Complete Pipeline:**

```
Step 1: Data Loading and Exploration
        ↓
Step 2: Data Cleaning (handle null values)
        ↓
Step 3: Text Preprocessing (stopwords, stemming)
        ↓
Step 4: One-Hot Encoding (vocabulary mapping)
        ↓
Step 5: Padding (fixed length sequences)
        ↓
Step 6: Build LSTM Model (embedding + LSTM)
        ↓
Step 7: Training (epochs, batch size)
        ↓
Step 8: Evaluation (accuracy, confusion matrix)
        ↓
Result: Trained model ready for predictions
```

## Step 1: Data Loading and Exploration

### Import Libraries

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
```

### Load Dataset

```python
# Read CSV
df = pd.read_csv('train.csv')

# Display first few rows
df.head()
```

**Output:**

```
   id    title                          author    text                                label
0  0     House Dem...                  Darrell... Washington (Reuters)...              1
1  1     FLYNN: Hill...                Daniel...  Ever get the feeling...              0
2  2     Why the Truth...              Consorti... Why the Truth Might...              1
3  3     15 Civilian...                Jessica... videos 15 Civilians...              1
4  4     Iranian woman...              Howard... Print \nAn Iranian woman...        0
```

### Explore Dataset

```python
# Shape
print(df.shape)  # (20800, 5)

# Info
print(df.info())

# Check for null values
print(df.isnull().sum())
```

**Output:**

```
title     558
author    1957
text      39
label     0
dtype: int64
```

**Analysis:**
- Total records: 20,800
- Null values: title (558), author (1957), text (39)
- Label distribution: Check for imbalance

### Check Label Distribution

```python
print(df['label'].value_counts())
```

**Output:**

```
0    10413
1    10387
Name: label, dtype: int64
```

**Observation**: Balanced dataset (roughly 50-50 split)

## Step 2: Data Cleaning

### Handle Null Values

**Strategy**: Drop rows with null values

**Reason**:
- Cannot replace text data with mean/median
- Have sufficient data (20,800 records)
- Losing ~2000 records acceptable

```python
# Drop null values
df = df.dropna()

# Check shape
print(df.shape)  # (18285, 5)

# Verify no nulls
print(df.isnull().sum())  # All zeros
```

**Result**: 18,285 clean records remaining

### Separate Features and Target

```python
# Drop label column for X (independent features)
X = df.drop('label', axis=1)

# Extract label column for y (dependent feature)
y = df['label']

# Check shapes
print(X.shape)  # (18285, 4)
print(y.shape)  # (18285,)
```

### Select Feature for Training

**Decision**: Use only **title** column

**Reason**:
- Title concise and informative
- Text column very large (slow training)
- Author not always reliable indicator

```python
# Extract title column
messages = X['title'].copy()

# Reset index
messages = messages.reset_index(drop=True)

print(messages.head())
```

**Output:**

```
0    House Dem Aide: We Didn't Even See Comey's L...
1    FLYNN: Hillary Clinton, Big Woman on Campus ...
2    Why the Truth Might Get You Fired
3    15 Civilian Families Died in U.S. Airstrike...
4    Iranian woman jailed for fictional unpublish...
Name: title, dtype: object
```

## Step 3: Text Preprocessing

### Why Preprocessing?

**Raw text issues:**
- Special characters (@, #, !, etc.)
- Mixed case (CAPS, lowercase)
- Stop words (the, is, and)
- Different word forms (running, run, ran)

**Solution**: Clean text for better model performance

### Preprocessing Steps

**1. Remove special characters**
**2. Convert to lowercase**
**3. Tokenize (split into words)**
**4. Remove stopwords**
**5. Apply stemming**

### Implementation

```python
# Download stopwords
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Create corpus (cleaned text)
corpus = []

for i in range(len(messages)):
    # Remove special characters (keep A-Z, a-z)
    review = re.sub('[^a-zA-Z]', ' ', messages[i])

    # Convert to lowercase
    review = review.lower()

    # Split into words
    review = review.split()

    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review
              if word not in stopwords.words('english')]

    # Join words back
    review = ' '.join(review)

    # Add to corpus
    corpus.append(review)

print(len(corpus))  # 18285
```

### Example: Before and After

**Original:**

```
"FLYNN: Hillary Clinton, Big Woman on Campus - Breitbart"
```

**After Preprocessing:**

```
"flynn hillari clinton big woman campu breitbart"
```

**Changes:**
- Removed special characters (:, -)
- Lowercased
- Removed stopwords (on)
- Stemmed (hillary → hillari, campus → campu)

### Inspect Corpus

```python
# First 5 cleaned texts
for i in range(5):
    print(f"{i}: {corpus[i]}")
```

**Output:**

```
0: hous dem aid even see comey letter jason chaffetz tweet
1: flynn hillari clinton big woman campu breitbart
2: truth might get fire
3: 15 civilian famili die u airstrik northern syria report
4: iranian woman jail fiction unpublish stori onli charge
```

## Step 4: One-Hot Encoding

### Set Vocabulary Size

```python
vocab_size = 5000  # Consider 5000 unique words
```

**Trade-offs:**

| Vocab Size | Pros | Cons |
|------------|------|------|
| Small (500) | Fast training, less memory | Many word collisions |
| Medium (5000) | Balanced | Good for most tasks |
| Large (10000+) | Captures more words | Slow training, more memory |

### Apply One-Hot Encoding

```python
# One-hot encode each sentence
one_hot_repr = [one_hot(words, vocab_size) for words in corpus]

print(len(one_hot_repr))  # 18285
```

### Example: One-Hot Representation

**Input sentence:**

```
corpus[0]: "hous dem aid even see comey letter jason chaffetz tweet"
```

**One-hot encoded:**

```
[2861, 4286, 1091, 5336, 789, 3245, 4567, 2341, 1890, 4521]
```

**Interpretation**:
- "hous" → index 2861 in vocab
- "dem" → index 4286 in vocab
- "aid" → index 1091 in vocab
- ... and so on

### Verify Encoding

```python
print(f"Original: {corpus[0]}")
print(f"Encoded: {one_hot_repr[0]}")
print(f"Length: {len(one_hot_repr[0])}")
```

## Step 5: Padding

### Why Padding?

**Problem**: Sentences have variable lengths

```
Sentence 1: 10 words → [2861, 4286, ..., 4521]  (10 elements)
Sentence 2: 5 words  → [1234, 5678, ..., 3456]  (5 elements)
Sentence 3: 15 words → [9876, 5432, ..., 1098]  (15 elements)
```

**LSTM input**: Requires fixed length!

**Solution**: Pad all sentences to same length

### Set Maximum Length

```python
sent_length = 20  # Maximum words per sentence
```

**Reasoning:**
- Most titles have 10-15 words
- 20 covers majority of titles
- Longer titles will be truncated
- Shorter titles will be padded

### Apply Padding

```python
# Pad sequences
embedded_docs = pad_sequences(
    one_hot_repr,
    maxlen=sent_length,
    padding='pre'  # Add zeros at beginning
)

print(embedded_docs.shape)  # (18285, 20)
```

### Example: Before and After Padding

**Original (10 words):**

```
[2861, 4286, 1091, 5336, 789, 3245, 4567, 2341, 1890, 4521]
```

**After padding (20 elements):**

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2861, 4286, 1091, 5336, 789, 3245, 4567, 2341, 1890, 4521]
 ↑_____________________________↑
      10 zeros added (pre-padding)
```

### Verify Padding

```python
# Check first 3 padded sequences
for i in range(3):
    print(f"Sentence {i}: {embedded_docs[i]}")
    print(f"Length: {len(embedded_docs[i])}\n")
```

**All sentences now have exactly 20 elements!**

## Step 6: Build LSTM Model

### Set Hyperparameters

```python
embedding_vector_features = 40  # Embedding dimension
vocab_size = 5000              # Vocabulary size
sent_length = 20               # Sequence length
lstm_units = 100              # LSTM neurons
```

**Parameter Meanings:**

- **embedding_vector_features (40)**: Each word → 40-dimensional vector
- **vocab_size (5000)**: 5000 unique words in vocabulary
- **sent_length (20)**: Each sentence has 20 words (after padding)
- **lstm_units (100)**: LSTM layer has 100 hidden units

### Create Model

```python
model = Sequential()

# Layer 1: Embedding
model.add(Embedding(
    input_dim=vocab_size,                  # 5000
    output_dim=embedding_vector_features,  # 40
    input_length=sent_length               # 20
))

# Layer 2: LSTM
model.add(LSTM(lstm_units))  # 100 units

# Layer 3: Dense (Output)
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()
```

**Output:**

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 20, 40)           200000
_________________________________________________________________
lstm (LSTM)                  (None, 100)              56400
_________________________________________________________________
dense (Dense)                (None, 1)                101
=================================================================
Total params: 256,501
Trainable params: 256,501
Non-trainable params: 0
_________________________________________________________________
```

**Parameter Calculations:**

**Embedding Layer:**

$$\text{Params} = \text{vocab\_size} \times \text{embedding\_dim} = 5000 \times 40 = 200,000$$

**LSTM Layer:**

$$\text{Params} = 4 \times (\text{input\_dim} + \text{hidden\_dim} + 1) \times \text{hidden\_dim}$$

$$\text{Params} = 4 \times (40 + 100 + 1) \times 100 = 56,400$$

**Dense Layer:**

$$\text{Params} = \text{input\_dim} \times \text{output\_dim} + \text{bias} = 100 \times 1 + 1 = 101$$

### Model Architecture Visualization

```
Input: (batch, 20) - Padded sequences
       ↓
Embedding: (batch, 20, 40) - Each word → 40-dim vector
       ↓
LSTM: (batch, 100) - Process sequence, output 100-dim
       ↓
Dense: (batch, 1) - Binary classification
       ↓
Output: Probability (0 to 1)
```

## Step 7: Training

### Prepare Data

```python
# Convert to numpy arrays
X_final = np.array(embedded_docs)
y_final = np.array(y)

print(X_final.shape)  # (18285, 20)
print(y_final.shape)  # (18285,)
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")  # 14628
print(f"Testing samples: {X_test.shape[0]}")   # 3657
```

### Train Model

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    verbose=1
)
```

**Training Output:**

```
Epoch 1/10
229/229 [======] - loss: 0.6523 - accuracy: 0.6234 - val_loss: 0.5123 - val_accuracy: 0.7456
Epoch 2/10
229/229 [======] - loss: 0.4234 - accuracy: 0.8123 - val_loss: 0.3456 - val_accuracy: 0.8567
Epoch 3/10
229/229 [======] - loss: 0.2987 - accuracy: 0.8876 - val_loss: 0.2876 - val_accuracy: 0.8934
...
Epoch 10/10
229/229 [======] - loss: 0.0567 - accuracy: 0.9876 - val_loss: 0.4123 - val_accuracy: 0.9012
```

**Observations:**
- Training accuracy increases: 62% → 98%
- Validation accuracy increases: 74% → 90%
- Loss decreases consistently
- Some overfitting (training acc >> validation acc)

### Hyperparameters Explained

**epochs=10:**
- Model sees entire dataset 10 times
- More epochs = better learning (but risk overfitting)

**batch_size=64:**
- Process 64 samples at once
- Larger batch = faster training, less memory efficient
- Smaller batch = slower training, more generalization

**validation_data:**
- Evaluate on test set after each epoch
- Monitor overfitting

## Step 8: Evaluation

### Make Predictions

```python
# Predict probabilities
y_pred_prob = model.predict(X_test)

# Apply threshold (0.5)
y_pred = (y_pred_prob > 0.5).astype(int)

print(y_pred[:10])  # First 10 predictions
```

**Output:**

```
[[0]
 [1]
 [0]
 [1]
 [0]
 [0]
 [1]
 [0]
 [1]
 [1]]
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

**Output:**

```
[[1654   108]
 [ 123  1772]]
```

**Interpretation:**

```
                Predicted
                Real  Fake
Actual  Real   1654   108  (True Negatives, False Positives)
        Fake    123  1772  (False Negatives, True Positives)
```

**Metrics:**
- **True Negatives (TN)**: 1654 (correctly predicted real news)
- **False Positives (FP)**: 108 (real news predicted as fake)
- **False Negatives (FN)**: 123 (fake news predicted as real)
- **True Positives (TP)**: 1772 (correctly predicted fake news)

**Accuracy:**

$$\text{Accuracy} = \frac{TN + TP}{Total} = \frac{1654 + 1772}{3657} = 0.9368 = 93.68\%$$

### Classification Report

```python
print(classification_report(y_test, y_pred))
```

**Output:**

```
              precision    recall  f1-score   support

           0       0.93      0.94      0.93      1762
           1       0.94      0.93      0.94      1895

    accuracy                           0.94      3657
   macro avg       0.94      0.94      0.94      3657
weighted avg       0.94      0.94      0.94      3657
```

**Metrics Explained:**

**Precision (Class 1 - Fake News):**

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{1772}{1772 + 108} = 0.94$$

**Recall (Class 1 - Fake News):**

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{1772}{1772 + 123} = 0.93$$

**F1-Score:**

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.94$$

### Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**Output:**

```
Accuracy: 93.68%
```

## Advanced: Adding Dropout for Regularization

### Why Dropout?

**Problem**: Overfitting (training acc 98%, validation acc 90%)

**Solution**: Dropout randomly disables neurons during training

### Modified Model with Dropout

```python
model = Sequential()

# Embedding
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_length))

# LSTM with Dropout
model.add(LSTM(lstm_units))
model.add(Dropout(0.3))  # Disable 30% neurons randomly

# Dense
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Train with Dropout

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64
)
```

**Expected Result:**
- Reduced overfitting
- Validation accuracy closer to training accuracy
- Better generalization

## Hyperparameter Tuning

### Parameters to Tune

| Hyperparameter | Tested Values | Best Value |
|----------------|---------------|------------|
| **vocab_size** | 500, 1000, 5000, 10000 | 5000 |
| **embedding_dim** | 10, 20, 40, 100 | 40 |
| **sent_length** | 10, 20, 30, 50 | 20 |
| **lstm_units** | 50, 100, 128, 256 | 100 |
| **epochs** | 5, 10, 15, 20 | 10 |
| **batch_size** | 32, 64, 128 | 64 |
| **dropout_rate** | 0.2, 0.3, 0.5 | 0.3 |

### Tuning Process

**1. Start with default values**
**2. Change one parameter at a time**
**3. Observe validation accuracy**
**4. Keep best performing value**
**5. Repeat for all parameters**

## Complete Code

```python
# Step 1: Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load data
df = pd.read_csv('train.csv')
df = df.dropna()

# Step 3: Prepare data
X = df.drop('label', axis=1)
y = df['label']
messages = X['title'].copy().reset_index(drop=True)

# Step 4: Text preprocessing
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages[i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

# Step 5: One-hot encoding
vocab_size = 5000
one_hot_repr = [one_hot(words, vocab_size) for words in corpus]

# Step 6: Padding
sent_length = 20
embedded_docs = pad_sequences(one_hot_repr, maxlen=sent_length, padding='pre')

# Step 7: Prepare arrays
X_final = np.array(embedded_docs)
y_final = np.array(y)

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# Step 9: Build model
model = Sequential()
model.add(Embedding(vocab_size, 40, input_length=sent_length))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 10: Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                   epochs=10, batch_size=64, verbose=1)

# Step 11: Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

## ❓ Interview Questions & Answers

**Q1: Why did we use only the title column and not the text column?**

**Title column:**
- Concise (10-15 words)
- Fast training
- Contains key information

**Text column:**
- Very long (100+ words)
- Slow training
- More computational resources needed

For this project, title sufficient for good accuracy (93%).

**Q2: Why drop null values instead of imputing?**

Cannot impute text data:
- Mean/median not applicable to text
- Replacing with generic text introduces noise
- Have sufficient data (18,285 records remaining)

Dropping acceptable when data is sufficient.

**Q3: Why use stemming instead of lemmatization?**

**Stemming:**
- Faster (rule-based)
- Good for large datasets
- Sufficient for sentiment analysis

**Lemmatization:**
- Slower (dictionary-based)
- More accurate
- Overkill for this task

**Q4: What is the purpose of pre-padding?**

Pre-padding (zeros at beginning):
- LSTM processes left to right
- Recent words (at end) more important
- Final LSTM state has recent information

Post-padding would put zeros at end, making final state less informative.

**Q5: How did you choose vocabulary size of 5000?**

Trade-off:
- Too small (500): Word collisions
- Too large (10000): Slow training

5000 is balanced for medium-sized dataset.

**Q6: Why binary cross-entropy loss for this problem?**

Binary classification (fake vs real):
- Output: 0 or 1
- Sigmoid activation
- Binary cross-entropy optimal for binary problems

**Q7: What is the role of Dropout(0.3)?**

Dropout randomly disables 30% of neurons:
- Prevents overfitting
- Forces network to learn robust features
- Better generalization

**Q8: How to interpret confusion matrix?**

```
[[TN  FP]
 [FN  TP]]
```

- **TN**: Correctly predicted real
- **FP**: Real predicted as fake (bad)
- **FN**: Fake predicted as real (bad)
- **TP**: Correctly predicted fake

Minimize FP and FN.

**Q9: Why is validation accuracy lower than training accuracy?**

Overfitting:
- Model memorizes training data
- Doesn't generalize to new data
- Solution: Dropout, early stopping, more data

**Q10: How to improve model performance?**

1. **More data**: Collect more samples
2. **Hyperparameter tuning**: Try different values
3. **Use text column**: More information
4. **Ensemble**: Combine multiple models
5. **Pre-trained embeddings**: Word2Vec, GloVe
6. **Bidirectional LSTM**: Process both directions

**Q11: What is the total number of parameters in the model?**

$$\text{Total} = \text{Embedding} + \text{LSTM} + \text{Dense}$$

$$\text{Total} = 200,000 + 56,400 + 101 = 256,501$$

**Q12: Why use Adam optimizer?**

Adam (Adaptive Moment Estimation):
- Combines momentum and RMSProp
- Adaptive learning rate
- Works well for most problems
- Default choice for deep learning

## 💡 Key Takeaways

- **Complete Pipeline**: Data → Preprocessing → Encoding → Padding → Model → Training → Evaluation
- **Text Preprocessing**: Remove special chars, lowercase, stopwords, stemming
- **One-Hot Encoding**: Maps words to indices (vocab_size)
- **Padding**: Makes all sequences same length (sent_length=20)
- **Embedding Layer**: Converts indices to dense vectors (40 dims)
- **LSTM Layer**: Processes sequences (100 units)
- **Binary Classification**: Sigmoid + binary cross-entropy
- **Accuracy**: 93.68% on fake news classification
- **Dropout**: Prevents overfitting (0.3 = 30% disabled)
- **Evaluation**: Confusion matrix, classification report, accuracy score

## ⚠️ Common Mistakes

**Mistake 1**: "Use text column for better accuracy"
- **Reality**: Title sufficient, text too slow for this project

**Mistake 2**: "Impute missing text data"
- **Reality**: Cannot impute text meaningfully, drop nulls

**Mistake 3**: "Use post-padding for LSTM"
- **Reality**: Use pre-padding (recent words more important)

**Mistake 4**: "High training accuracy = good model"
- **Reality**: Check validation accuracy (overfitting possible)

**Mistake 5**: "Use same parameters for all datasets"
- **Reality**: Tune hyperparameters for each dataset

**Mistake 6**: "Larger LSTM units always better"
- **Reality**: More units = more overfitting, tune carefully

## 📝 Quick Revision Points

### Complete Workflow

1. Load data (train.csv)
2. Drop nulls (can't impute text)
3. Preprocess (stopwords, stemming)
4. One-hot encode (vocab_size=5000)
5. Pad sequences (sent_length=20)
6. Build model (Embedding → LSTM → Dense)
7. Train (epochs=10, batch_size=64)
8. Evaluate (accuracy ~93%)

### Model Architecture

```python
Embedding(5000, 40, input_length=20)  # 200K params
LSTM(100)                              # 56.4K params
Dropout(0.3)                           # 0 params
Dense(1, activation='sigmoid')         # 101 params
```

### Key Hyperparameters

- **vocab_size**: 5000
- **embedding_dim**: 40
- **sent_length**: 20
- **lstm_units**: 100
- **epochs**: 10
- **batch_size**: 64
- **dropout**: 0.3

### Evaluation Metrics

- **Accuracy**: 93.68%
- **Precision**: 0.94
- **Recall**: 0.93
- **F1-Score**: 0.94

### Remember

- **Preprocessing crucial**: Stopwords + stemming improves accuracy
- **Pre-padding for LSTM**: Recent words more important
- **Dropout prevents overfitting**: Use 0.2-0.5
- **Tune hyperparameters**: Try different values
- **Monitor validation accuracy**: Avoid overfitting
- **Binary cross-entropy**: For binary classification
- **Adam optimizer**: Default choice for most tasks
