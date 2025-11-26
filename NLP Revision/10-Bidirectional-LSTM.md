# Chapter 10: Bidirectional LSTM

## 🎯 Learning Objectives
- Understand limitations of standard LSTM
- Learn bidirectional LSTM concept and architecture
- Master forward and backward processing
- Implement bidirectional LSTM in Keras
- Apply to real-world text classification problems
- Know when to use bidirectional vs unidirectional LSTM

## 📚 Key Concepts

### Problem with Standard LSTM

#### Limitation: Only Past Context

**Example Sentence:**

```
"Krish likes to eat _____ in Bangalore"
```

**Prediction Task**: Fill in the blank

**Standard LSTM Processing (Left to Right):**

```
Time Step 1: "Krish"
Time Step 2: "likes"
Time Step 3: "to"
Time Step 4: "eat"
Time Step 5: "_____"  ← Predict this word
```

**At prediction (Time Step 5):**

**Has context of**:
- "Krish" (previous)
- "likes" (previous)
- "to" (previous)
- "eat" (previous)

**Missing context of**:
- "in" (future)
- "Bangalore" (future)

#### Why Future Context Matters

**Scenario 1:**

```
"Krish likes to eat _____ in Bangalore"
```

Knowing "Bangalore" helps predict: **"dosa"** or **"biryani"** (famous foods in Bangalore)

**Scenario 2:**

```
"Krish likes to eat _____ in Lucknow"
```

Knowing "Lucknow" helps predict: **"tunday kebab"** (famous food in Lucknow)

**Conclusion**: Future words provide crucial context for accurate prediction!

### Standard LSTM Architecture Review

**Unidirectional LSTM (Standard):**

```
Input:    x₁      x₂      x₃      x₄      x₅
           ↓       ↓       ↓       ↓       ↓
       ┌──────┬──────┬──────┬──────┬──────┐
       │LSTM  │LSTM  │LSTM  │LSTM  │LSTM  │
       └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
          ↓      ↓      ↓      ↓      ↓
Output:  y₁     y₂     y₃     y₄     y₅
```

**Information Flow:**

$$h_1 = f(x_1, h_0)$$

$$h_2 = f(x_2, h_1)$$

$$h_3 = f(x_3, h_2)$$

$$h_4 = f(x_4, h_3)$$

$$h_5 = f(x_5, h_4)$$

**At each time step $t$:**
- Has information from $x_1$ to $x_t$ (past)
- Missing information from $x_{t+1}$ to $x_n$ (future)

### Bidirectional LSTM Concept

#### Core Idea

**Process sequence in BOTH directions:**
1. **Forward LSTM**: Left to right (past context)
2. **Backward LSTM**: Right to left (future context)

**Combine outputs** to get complete context

#### Bidirectional LSTM Architecture

**Visual Representation:**

```
Forward Direction →
Input:    x₁      x₂      x₃      x₄      x₅
           ↓       ↓       ↓       ↓       ↓
       ┌──────┬──────┬──────┬──────┬──────┐
       │LSTM→ │LSTM→ │LSTM→ │LSTM→ │LSTM→ │ Forward
       └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
          ↓      ↓      ↓      ↓      ↓
          ⊕      ⊕      ⊕      ⊕      ⊕     Combine
          ↑      ↑      ↑      ↑      ↑
       ┌──┴───┬──┴───┬──┴───┬──┴───┬──┴───┐
       │LSTM← │LSTM← │LSTM← │LSTM← │LSTM← │ Backward
       └──────┴──────┴──────┴──────┴──────┘
           ↑       ↑       ↑       ↑       ↑
Input:    x₁      x₂      x₃      x₄      x₅
← Backward Direction
```

**Symbol ⊕**: Concatenation (combine forward and backward outputs)

#### Information Flow

**Forward LSTM (→):**

$$\overrightarrow{h}_1 = f(x_1, \overrightarrow{h}_0)$$

$$\overrightarrow{h}_2 = f(x_2, \overrightarrow{h}_1)$$

$$\overrightarrow{h}_3 = f(x_3, \overrightarrow{h}_2)$$

$$\overrightarrow{h}_4 = f(x_4, \overrightarrow{h}_3)$$

$$\overrightarrow{h}_5 = f(x_5, \overrightarrow{h}_4)$$

**Backward LSTM (←):**

$$\overleftarrow{h}_5 = f(x_5, \overleftarrow{h}_6)$$

$$\overleftarrow{h}_4 = f(x_4, \overleftarrow{h}_5)$$

$$\overleftarrow{h}_3 = f(x_3, \overleftarrow{h}_4)$$

$$\overleftarrow{h}_2 = f(x_2, \overleftarrow{h}_3)$$

$$\overleftarrow{h}_1 = f(x_1, \overleftarrow{h}_2)$$

**Combined Output:**

$$h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

(Concatenation of forward and backward hidden states)

#### Example: At Time Step 3

**At $t=3$ (predicting for "eat"):**

**Forward LSTM** ($\overrightarrow{h}_3$):
- Has context from: $x_1$ ("Krish"), $x_2$ ("likes"), $x_3$ ("to")
- Knows: Previous words

**Backward LSTM** ($\overleftarrow{h}_3$):
- Has context from: $x_5$ ("Bangalore"), $x_4$ ("in")
- Knows: Future words

**Combined** ($h_3 = [\overrightarrow{h}_3 ; \overleftarrow{h}_3]$):
- Has context from BOTH previous AND future words
- Complete sentence understanding!

### Advantages of Bidirectional LSTM

**1. Complete Context:**
- Past AND future information
- Better understanding of word meaning

**2. Better Accuracy:**
- More informed predictions
- Reduces ambiguity

**3. Ideal for:**
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Sentiment analysis
- Text classification

**4. Not Time-Dependent:**
- When entire sequence available at once
- Batch processing acceptable

### Disadvantages of Bidirectional LSTM

**1. Cannot Use for:**
- Real-time prediction
- Online learning
- Streaming data

**2. Requires Complete Sequence:**
- Must have entire sentence
- Cannot process word-by-word in real-time

**3. More Parameters:**
- 2× parameters (forward + backward)
- Slower training
- More memory

**4. Not Suitable for:**
- Text generation
- Machine translation (encoder can be bidirectional, decoder cannot)
- Speech recognition (real-time)

### When to Use Bidirectional vs Unidirectional

| Use Case | Unidirectional LSTM | Bidirectional LSTM |
|----------|---------------------|---------------------|
| **Sentiment Analysis** | ✗ | ✓ (Complete sentence available) |
| **Text Classification** | ✗ | ✓ (Complete document available) |
| **Named Entity Recognition** | ✗ | ✓ (Context crucial) |
| **Text Generation** | ✓ | ✗ (Cannot see future) |
| **Machine Translation** | Decoder: ✓ | Encoder: ✓, Decoder: ✗ |
| **Speech Recognition (real-time)** | ✓ | ✗ (Streaming) |
| **Time Series Forecasting** | ✓ | ✗ (Cannot see future) |

## Implementation in Keras

### Standard LSTM vs Bidirectional LSTM

**Standard LSTM:**

```python
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(100))  # 100 units
model.add(Dense(1, activation='sigmoid'))
```

**Output shape after LSTM**: (batch_size, 100)

**Bidirectional LSTM:**

```python
from tensorflow.keras.layers import LSTM, Bidirectional

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(Bidirectional(LSTM(100)))  # 100 units × 2 directions
model.add(Dense(1, activation='sigmoid'))
```

**Output shape after Bidirectional LSTM**: (batch_size, 200)
- Forward LSTM: 100 units
- Backward LSTM: 100 units
- Concatenated: 200 units total

### Parameter Count Comparison

**Standard LSTM(100):**

$$\text{Params} = 4 \times (100 + \text{input\_dim} + 1) \times 100$$

**Bidirectional(LSTM(100)):**

$$\text{Params} = 2 \times [4 \times (100 + \text{input\_dim} + 1) \times 100]$$

**Bidirectional has 2× parameters of standard LSTM**

### Complete Example: Fake News Classifier

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
vocab_size = 5000
embedding_dim = 40
max_length = 20
lstm_units = 100

# Build model
model = Sequential()

# Embedding layer
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
))

# Bidirectional LSTM layer
model.add(Bidirectional(LSTM(lstm_units)))  # 100 forward + 100 backward

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
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
bidirectional (Bidirectional)(None, 200)              112800
_________________________________________________________________
dense (Dense)                (None, 1)                201
=================================================================
Total params: 313,001
Trainable params: 313,001
Non-trainable params: 0
_________________________________________________________________
```

**Key Observations:**
- Bidirectional output: (None, 200) instead of (None, 100)
- 2× parameters in Bidirectional layer
- Total params: 313,001

### Training the Model

```python
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=0.2,
    random_state=42
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)
```

**Expected Output:**

```
Epoch 1/10
- loss: 0.6234 - accuracy: 0.6456 - val_loss: 0.4892 - val_accuracy: 0.7678
Epoch 2/10
- loss: 0.3987 - accuracy: 0.8234 - val_loss: 0.3456 - val_accuracy: 0.8567
...
Epoch 10/10
- loss: 0.0678 - accuracy: 0.9845 - val_loss: 0.3892 - val_accuracy: 0.9012
```

### Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification report
print(classification_report(y_test, y_pred))
```

**Expected Accuracy**: ~90-92% (slightly better than unidirectional LSTM)

## Advanced: Stacked Bidirectional LSTM

### Multiple Bidirectional Layers

```python
model = Sequential()

# Embedding
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

# First Bidirectional LSTM (return sequences for stacking)
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# Second Bidirectional LSTM
model.add(Bidirectional(LSTM(64)))

# Output
model.add(Dense(1, activation='sigmoid'))
```

**Architecture:**

```
Embedding: (batch, 20, 40)
       ↓
Bidirectional LSTM 1: (batch, 20, 256)  ← 128×2 with sequences
       ↓
Bidirectional LSTM 2: (batch, 128)      ← 64×2
       ↓
Dense: (batch, 1)
```

**When to use:**
- Very complex tasks
- Large datasets
- Need deeper representations

## Practical Application: Complete Project

### Problem: Fake News Classification

**Dataset**: Kaggle Fake News Dataset
- **Features**: title, author, text
- **Target**: label (0=real, 1=fake)

### Step 1: Data Preprocessing

```python
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load data
df = pd.read_csv('train.csv', engine='python', error_bad_lines=False)

# Drop nulls
df = df.dropna()

# Separate features
X = df.drop('label', axis=1)
y = df['label']

# Use title column
messages = X['title'].copy().reset_index(drop=True)

# Preprocessing
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    # Remove special characters
    review = re.sub('[^a-zA-Z]', ' ', messages[i])

    # Lowercase and split
    review = review.lower().split()

    # Remove stopwords and stem
    review = [ps.stem(word) for word in review
              if word not in stopwords.words('english')]

    # Join back
    corpus.append(' '.join(review))
```

### Step 2: One-Hot Encoding and Padding

```python
# One-hot encoding
vocab_size = 5000
one_hot_repr = [one_hot(words, vocab_size) for words in corpus]

# Padding
max_length = 20
embedded_docs = pad_sequences(
    one_hot_repr,
    maxlen=max_length,
    padding='pre'
)

# Prepare arrays
X_final = np.array(embedded_docs)
y_final = np.array(y)
```

### Step 3: Build Bidirectional LSTM Model

```python
model = Sequential()

# Embedding
model.add(Embedding(vocab_size, 40, input_length=max_length))

# Bidirectional LSTM
model.add(Bidirectional(LSTM(100)))

# Output
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Step 4: Train and Evaluate

```python
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**Expected Result**: ~90% accuracy

## ❓ Interview Questions & Answers

**Q1: What is Bidirectional LSTM?**

Bidirectional LSTM processes sequence in both directions:
- **Forward LSTM**: Left to right (past context)
- **Backward LSTM**: Right to left (future context)
- **Output**: Concatenation of both directions

Each output has complete context (past + future).

**Q2: Why use Bidirectional LSTM over standard LSTM?**

**Standard LSTM**: Only past context
- At time t, knows x₁ to x_t
- Missing future information

**Bidirectional LSTM**: Past + future context
- At time t, knows x₁ to x_n (complete sequence)
- Better understanding, higher accuracy

**Q3: What is the output shape of Bidirectional LSTM?**

**Standard LSTM(100)**:
- Output shape: (batch, 100)

**Bidirectional(LSTM(100))**:
- Forward: 100 units
- Backward: 100 units
- Output shape: (batch, 200)  ← Concatenated

**Q4: How many parameters does Bidirectional LSTM have?**

**2× parameters of standard LSTM**

If LSTM(100) has P parameters:
- Bidirectional(LSTM(100)) has 2P parameters

**Q5: When NOT to use Bidirectional LSTM?**

**Cannot use for:**
1. **Text generation**: Cannot see future when generating
2. **Real-time prediction**: Requires complete sequence
3. **Streaming data**: Need to process incrementally
4. **Time series forecasting**: Cannot know future values

**Q6: Can we use Bidirectional LSTM for machine translation?**

**Yes, but only for encoder**:
- **Encoder**: Bidirectional (complete source sentence available)
- **Decoder**: Unidirectional (generating target word-by-word)

**Q7: What is the formula for combined output in Bidirectional LSTM?**

$$h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

Where:
- $\overrightarrow{h}_t$ = Forward LSTM output at time t
- $\overleftarrow{h}_t$ = Backward LSTM output at time t
- ; = Concatenation operation

**Q8: How to implement Bidirectional LSTM in Keras?**

```python
from tensorflow.keras.layers import Bidirectional, LSTM

model.add(Bidirectional(LSTM(100)))
```

Wraps standard LSTM layer with Bidirectional() wrapper.

**Q9: Can we stack multiple Bidirectional LSTM layers?**

Yes, use `return_sequences=True`:

```python
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
```

First layer returns sequences, second processes them.

**Q10: What is the training time difference?**

**Bidirectional LSTM**:
- 2× parameters
- ~1.5-2× training time (not exactly 2× due to parallel processing)
- More memory required

**Trade-off**: Slower training but better accuracy

**Q11: Give an example where Bidirectional LSTM significantly helps.**

**Sentence**: "The food was not good"

**Standard LSTM** at "not":
- Context: "The food was not"
- May miss that "not" negates "good"

**Bidirectional LSTM** at "not":
- Forward: "The food was not"
- Backward: "good"
- Understands negation better!

**Q12: Can Bidirectional LSTM be used for Named Entity Recognition?**

**Yes, ideal for NER!**

Example: "John lives in Paris"

At "Paris":
- Forward: Knows "John lives in"
- Backward: (end of sentence)
- Combined: Understands "Paris" is location (after "in")

## 💡 Key Takeaways

- **Bidirectional LSTM** = Forward + Backward processing
- **Output**: Concatenation of both directions
- **Complete Context**: Past AND future information
- **Output Shape**: 2× standard LSTM (e.g., 200 instead of 100)
- **Parameters**: 2× standard LSTM
- **Use For**: Sentiment analysis, text classification, NER, POS tagging
- **NOT For**: Text generation, real-time prediction, streaming
- **Keras**: `Bidirectional(LSTM(units))`
- **Accuracy**: Typically 2-5% better than standard LSTM
- **Trade-off**: Slower training, better accuracy

## ⚠️ Common Mistakes

**Mistake 1**: "Use Bidirectional LSTM for text generation"
- **Reality**: Cannot use (cannot see future when generating)

**Mistake 2**: "Bidirectional LSTM output shape same as standard LSTM"
- **Reality**: 2× output size (concatenation of forward + backward)

**Mistake 3**: "Always use Bidirectional LSTM"
- **Reality**: Only when complete sequence available at once

**Mistake 4**: "Bidirectional LSTM for real-time speech recognition"
- **Reality**: Cannot use (requires complete sentence)

**Mistake 5**: "Bidirectional LSTM has same parameters as standard LSTM"
- **Reality**: 2× parameters (forward + backward layers)

**Mistake 6**: "Stack Bidirectional without return_sequences"
- **Reality**: First layer needs `return_sequences=True`

## 📝 Quick Revision Points

### Architecture

```
Forward  LSTM →  →  →  →  →
Input:    x₁  x₂  x₃  x₄  x₅
Backward LSTM ←  ←  ←  ←  ←

Output: Concatenation of both
```

### Code

```python
# Standard LSTM
model.add(LSTM(100))  # Output: (batch, 100)

# Bidirectional LSTM
model.add(Bidirectional(LSTM(100)))  # Output: (batch, 200)
```

### Output Formula

$$h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$$

### When to Use

| Task | Use Bidirectional? |
|------|-------------------|
| Sentiment Analysis | ✓ Yes |
| Text Classification | ✓ Yes |
| Named Entity Recognition | ✓ Yes |
| Text Generation | ✗ No |
| Real-time Prediction | ✗ No |

### Parameters

- **Standard LSTM(100)**: P parameters
- **Bidirectional(LSTM(100))**: 2P parameters

### Remember

- **Bidirectional** = Forward + Backward
- **Output** = 2× size (concatenation)
- **Parameters** = 2× count
- **Context** = Complete (past + future)
- **Use** = Complete sequence available
- **NOT** = Real-time, generation, streaming
