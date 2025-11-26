# Chapter 8: Word Embedding Layer and Practical Implementation

## 🎯 Learning Objectives
- Understand embedding layer in neural networks
- Master text preprocessing pipeline for deep learning
- Learn one-hot encoding with TensorFlow
- Understand padding (pre-padding and post-padding)
- Implement embedding layer in Keras
- Know when to use embedding layer vs Word2Vec
- Complete practical example with code

## 📚 Key Concepts

### Text Preprocessing Pipeline for Deep Learning

**Complete Pipeline:**

```
Step 1: Sentences (raw text)
        ↓
Step 2: One-Hot Encoding (get index positions)
        ↓
Step 3: Vocabulary Size (unique words count)
        ↓
Step 4: Padding (make all sentences same length)
        ↓
Step 5: Embedding Layer (convert to dense vectors)
        ↓
Result: Dense vector representations
```

**Why This Pipeline?**

Traditional methods (BoW, TF-IDF, Avg Word2Vec):
- Use pre-trained models or count-based methods
- Don't train embeddings with your specific task

**Embedding Layer Approach:**
- Trains embeddings as part of your neural network
- Task-specific embeddings
- End-to-end learning

### Step 1: Sentences (Raw Text)

**Example Dataset:**

```python
sentences = [
    "I have a glass of milk",
    "I have a glass of juice",
    "I have a cup of tea",
    "I am a good boy",
    "I am a good developer",
    "Understand the meaning of words",
    "Your videos are good"
]
```

**Analysis:**

```
Sentence 1: 6 words
Sentence 2: 6 words
Sentence 3: 6 words
Sentence 4: 5 words
Sentence 5: 5 words
Sentence 6: 5 words
Sentence 7: 4 words

Problem: Different lengths!
```

### Step 2: One-Hot Encoding

#### What is One-Hot Encoding?

**Concept**: Represent each word by its index in vocabulary

**Traditional One-Hot (Not Used Here):**

```
Vocabulary: [good, boy, girl]

"good" → [1, 0, 0]
"boy"  → [0, 1, 0]
"girl" → [0, 0, 1]
```

Problem: Sparse, high-dimensional

**Index-Based One-Hot (Used in Deep Learning):**

```
Vocabulary: [the, food, is, good, bad, ...]  (size = 10000)

"the"  → Index 2423 (position where 1 exists in 10000-dim vector)
"food" → Index 5188
"is"   → Index 1091
"good" → Index 5336
```

We only store the **index**, not the full sparse vector!

#### TensorFlow Implementation

**Import:**

```python
from tensorflow.keras.preprocessing.text import one_hot
```

**Usage:**

```python
vocab_size = 10000

# For each sentence, get index representation
encoded_sentences = []
for sentence in sentences:
    encoded = one_hot(sentence, vocab_size)
    encoded_sentences.append(encoded)
```

**Example Output:**

```python
Sentence: "I have a glass of milk"

One-hot encoded (indices):
[2423, 5188, 1091, 5336, 789, 3245]

Interpretation:
"I"     → Index 2423 in vocab
"have"  → Index 5188 in vocab
"a"     → Index 1091 in vocab
"glass" → Index 5336 in vocab
"of"    → Index 789 in vocab
"milk"  → Index 3245 in vocab
```

#### Example: Two Similar Sentences

**Sentences:**

```
S1: "I have a glass of milk"
S2: "I have a glass of juice"
```

**One-hot encoded:**

```
S1: [2423, 5188, 1091, 5336, 789, 3245]
S2: [2423, 5188, 1091, 5336, 789, 8901]
                                    ↑
                            Only difference!
```

**Observation**: Same context, only last word differs

### Step 3: Vocabulary Size

#### What is Vocabulary Size?

**Definition**: Number of unique words in your corpus

**Example:**

```
Sentences:
- "I have a glass of milk"
- "I have a glass of juice"
- "I have a cup of tea"

Unique words: {I, have, a, glass, of, milk, juice, cup, tea}
Vocabulary size: 9
```

#### Choosing Vocabulary Size

**For Real Applications:**

| Dataset Size | Recommended Vocab Size |
|--------------|------------------------|
| Small (< 1000 docs) | 500 - 1000 |
| Medium (1K - 10K docs) | 5000 - 10000 |
| Large (> 10K docs) | 20000 - 50000 |
| Very Large (100K+ docs) | 50000 - 100000 |

**Trade-offs:**

**Small Vocabulary (500):**
- ✓ Faster training
- ✓ Less memory
- ✗ May miss rare words
- ✗ More collisions (different words same index)

**Large Vocabulary (10000):**
- ✓ Fewer collisions
- ✓ Captures more words
- ✗ Slower training
- ✗ More memory

**Example:**

```python
vocab_size = 500  # Smaller, faster

# One-hot with 500 vocab
[142, 388, 91, 336, 89, 245]  # Indices range: 0-499

vocab_size = 10000  # Larger, more accurate

# One-hot with 10000 vocab
[2423, 5188, 1091, 5336, 789, 3245]  # Indices range: 0-9999
```

### Step 4: Padding

#### Why Padding?

**Problem**: Neural networks need fixed input size

```
Sentence 1: [2423, 5188, 1091, 5336, 789, 3245]  (6 words)
Sentence 2: [1234, 5678, 9012, 3456, 7890]      (5 words)
Sentence 3: [1111, 2222, 3333, 4444]            (4 words)

Neural Network Input: Must be same length!
```

**Solution**: Add zeros to make all sentences same length

#### Pre-Padding vs Post-Padding

**Pre-Padding (Add zeros at beginning):**

```
Original: [2423, 5188, 1091, 5336]  (4 words)
Max length: 8

Pre-padded: [0, 0, 0, 0, 2423, 5188, 1091, 5336]
             ↑________↑
          4 zeros added at start
```

**Post-Padding (Add zeros at end):**

```
Original: [2423, 5188, 1091, 5336]  (4 words)
Max length: 8

Post-padded: [2423, 5188, 1091, 5336, 0, 0, 0, 0]
                                      ↑________↑
                                  4 zeros added at end
```

#### When to Use Which?

**Pre-Padding (Most Common):**
- LSTM, GRU (processes left-to-right)
- Recent words more important
- Zeros at beginning don't affect final state

**Post-Padding:**
- Bidirectional RNN
- When word position matters

#### TensorFlow Implementation

**Import:**

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

**Pre-Padding:**

```python
max_length = 8

padded_sentences = pad_sequences(
    encoded_sentences,
    maxlen=max_length,
    padding='pre'  # Add zeros at beginning
)
```

**Post-Padding:**

```python
padded_sentences = pad_sequences(
    encoded_sentences,
    maxlen=max_length,
    padding='post'  # Add zeros at end
)
```

#### Complete Example

**Original Sentences:**

```python
S1: [2423, 5188, 1091, 5336]  (4 words)
S2: [1234, 5678, 9012, 3456, 7890]  (5 words)
S3: [1111, 2222, 3333, 4444, 5555, 6666]  (6 words)
```

**After Pre-Padding (max_length=8):**

```python
S1: [0, 0, 0, 0, 2423, 5188, 1091, 5336]  (8 elements)
S2: [0, 0, 0, 1234, 5678, 9012, 3456, 7890]  (8 elements)
S3: [0, 0, 1111, 2222, 3333, 4444, 5555, 6666]  (8 elements)

All sentences now have 8 elements!
```

**How Many Zeros Added:**

```
S1: 4 words → 8-4 = 4 zeros added
S2: 5 words → 8-5 = 3 zeros added
S3: 6 words → 8-6 = 2 zeros added
```

### Step 5: Embedding Layer

#### What is Embedding Layer?

**Purpose**: Convert word indices to dense vectors

**Input**: Word index (integer)
**Output**: Dense vector (e.g., 10 dimensions)

**Example:**

```
Input:  Index 2423 (represents "I")
        ↓
Embedding Layer (dimension=10)
        ↓
Output: [0.23, -0.45, 0.67, 0.12, -0.89, 0.34, -0.56, 0.78, 0.91, -0.23]
        ↑_____________________________________________________________↑
                           10-dimensional vector
```

#### Embedding Layer vs Word2Vec

| Aspect | Word2Vec | Embedding Layer |
|--------|----------|-----------------|
| **Training** | Pre-trained or separate | Part of neural network |
| **Task** | General purpose | Task-specific |
| **When to Use** | Limited data, general text | Large data, specific task |
| **Flexibility** | Fixed embeddings | Updates during training |
| **Example** | Google News 300D | Trained with your model |

**When to Use Embedding Layer:**
- You have your own dataset
- Task-specific embeddings needed
- Training end-to-end model
- Sufficient training data

**When to Use Word2Vec:**
- Small dataset
- General text
- Transfer learning
- Pre-trained model available

#### Embedding Layer in Keras

**Import:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
```

**Creating Embedding Layer:**

```python
model = Sequential()

model.add(Embedding(
    input_dim=vocab_size,      # Vocabulary size (e.g., 500)
    output_dim=embedding_dim,  # Embedding dimension (e.g., 10)
    input_length=max_length    # Max sentence length (e.g., 8)
))
```

**Parameters Explained:**

**1. input_dim (Vocabulary Size):**
- Number of unique words
- Example: 500, 10000

**2. output_dim (Embedding Dimension):**
- Size of dense vector for each word
- Example: 10, 50, 100, 300
- Similar to Word2Vec dimension

**3. input_length (Maximum Sentence Length):**
- Length after padding
- Example: 8, 100, 200

#### Complete Model Example

```python
vocab_size = 500
embedding_dim = 10
max_length = 8

model = Sequential()

# Embedding layer
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
))

# Compile model
model.compile(
    optimizer='adam',
    loss='mse'  # Or any loss function
)

# Model summary
model.summary()
```

**Output:**

```
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 8, 10)            5000
=================================================================
Total params: 5,000
Trainable params: 5,000
Non-trainable params: 0
```

**Parameter Calculation:**

$$\text{Params} = \text{vocab\_size} \times \text{embedding\_dim}$$

$$\text{Params} = 500 \times 10 = 5000$$

#### Getting Vector Representations

**Input Sentence:**

```python
sentence = [0, 0, 0, 0, 2423, 5188, 1091, 5336]  # Padded sentence
```

**Prediction:**

```python
vector = model.predict([sentence])

# Output shape: (1, 8, 10)
# 1 sentence, 8 words, 10 dimensions per word
```

**Result:**

```
Word 1 (index 0):    [0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34]
Word 2 (index 0):    [0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34]
Word 3 (index 0):    [0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34]
Word 4 (index 0):    [0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34]
Word 5 (index 2423): [0.45, -0.23, 0.67, 0.12, -0.89, 0.34, 0.56, -0.78, 0.91, 0.23]
Word 6 (index 5188): [0.23, 0.56, -0.34, 0.78, 0.12, -0.45, 0.67, 0.89, -0.23, 0.56]
Word 7 (index 1091): [-0.34, 0.67, 0.23, -0.56, 0.89, 0.12, -0.78, 0.45, 0.67, -0.23]
Word 8 (index 5336): [0.67, -0.12, 0.45, 0.23, -0.67, 0.89, 0.34, -0.56, 0.78, 0.12]
```

**Observation**: Same indices (0) have same vectors, different indices have different vectors

### Complete Practical Example

#### Step-by-Step Implementation

**Step 1: Import Libraries**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
```

**Step 2: Define Sentences**

```python
sentences = [
    "I have a glass of milk",
    "I have a glass of juice",
    "I have a cup of tea",
    "I am a good boy",
    "I am a good developer",
    "Understand the meaning of words",
    "Your videos are good"
]
```

**Step 3: Set Parameters**

```python
vocab_size = 500        # Vocabulary size
embedding_dim = 10      # Embedding dimension
max_length = 8          # Maximum sentence length
```

**Step 4: One-Hot Encoding**

```python
encoded_sentences = []
for sentence in sentences:
    encoded = one_hot(sentence, vocab_size)
    encoded_sentences.append(encoded)

print(encoded_sentences)
```

**Output:**

```
[
    [142, 388, 91, 336, 89, 245],
    [142, 388, 91, 336, 89, 401],
    [142, 388, 91, 123, 89, 456],
    [142, 234, 91, 178, 321],
    [142, 234, 91, 178, 467],
    [289, 423, 156, 89, 234],
    [390, 234, 167, 178]
]
```

**Step 5: Padding**

```python
padded_sentences = pad_sequences(
    encoded_sentences,
    maxlen=max_length,
    padding='pre'
)

print(padded_sentences)
```

**Output:**

```
[
    [0, 0, 142, 388, 91, 336, 89, 245],
    [0, 0, 142, 388, 91, 336, 89, 401],
    [0, 0, 142, 388, 91, 123, 89, 456],
    [0, 0, 0, 142, 234, 91, 178, 321],
    [0, 0, 0, 142, 234, 91, 178, 467],
    [0, 0, 0, 289, 423, 156, 89, 234],
    [0, 0, 0, 0, 390, 234, 167, 178]
]
```

**Step 6: Create Embedding Layer**

```python
model = Sequential()

model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
))

model.compile(optimizer='adam', loss='mse')

model.summary()
```

**Output:**

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 8, 10)            5000
=================================================================
Total params: 5,000
Trainable params: 5,000
Non-trainable params: 0
_________________________________________________________________
```

**Step 7: Get Vector Representations**

```python
# For first sentence
vectors = model.predict(padded_sentences[0:1])

print("Shape:", vectors.shape)  # (1, 8, 10)
print("\nVectors:")
print(vectors)
```

**Output:**

```
Shape: (1, 8, 10)

Vectors:
[[[0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34],
  [0.12, 0.34, -0.56, 0.78, 0.23, -0.45, 0.67, 0.89, -0.12, 0.34],
  [0.45, -0.23, 0.67, 0.12, -0.89, 0.34, 0.56, -0.78, 0.91, 0.23],
  [0.23, 0.56, -0.34, 0.78, 0.12, -0.45, 0.67, 0.89, -0.23, 0.56],
  [-0.34, 0.67, 0.23, -0.56, 0.89, 0.12, -0.78, 0.45, 0.67, -0.23],
  [0.67, -0.12, 0.45, 0.23, -0.67, 0.89, 0.34, -0.56, 0.78, 0.12],
  [0.89, 0.23, -0.45, 0.67, 0.34, -0.78, 0.56, 0.23, -0.89, 0.45],
  [0.34, -0.56, 0.78, 0.12, -0.34, 0.67, 0.23, -0.89, 0.56, 0.78]]]
```

**Step 8: Get Vectors for All Sentences**

```python
all_vectors = model.predict(padded_sentences)

print("Shape:", all_vectors.shape)  # (7, 8, 10)
# 7 sentences, 8 words each, 10 dimensions per word
```

### Using Embedding Layer in LSTM Model

**Complete Model with LSTM:**

```python
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()

# Embedding layer
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
))

# LSTM layer
model.add(LSTM(128))  # 128 hidden units

# Dense output layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Output:**

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 8, 10)            5000
_________________________________________________________________
lstm (LSTM)                  (None, 128)              71168
_________________________________________________________________
dense (Dense)                (None, 1)                129
=================================================================
Total params: 76,297
Trainable params: 76,297
Non-trainable params: 0
_________________________________________________________________
```

**Information Flow:**

```
Input: [0, 0, 0, 0, 2423, 5188, 1091, 5336]  (padded sentence)
       ↓
Embedding Layer: (batch, 8, 10)
       ↓ Each word → 10-dim vector
LSTM Layer: (batch, 128)
       ↓ Processes sequence
Dense Layer: (batch, 1)
       ↓ Final prediction
Output: 0.87 (probability)
```

## ❓ Interview Questions & Answers

**Q1: What is the embedding layer in neural networks?**

Embedding layer converts word indices (integers) to dense vectors:
- Input: Word index (e.g., 2423)
- Output: Dense vector (e.g., 10 dimensions)
- Trainable: Updates during model training
- Part of neural network (end-to-end learning)

**Q2: What are the parameters of Embedding layer?**

```python
Embedding(input_dim, output_dim, input_length)
```

- **input_dim**: Vocabulary size (e.g., 500, 10000)
- **output_dim**: Embedding dimension (e.g., 10, 100, 300)
- **input_length**: Maximum sentence length (e.g., 8, 100)

**Q3: What is the difference between pre-padding and post-padding?**

**Pre-padding**: Add zeros at beginning
```
[2423, 5188, 1091] → [0, 0, 0, 0, 0, 2423, 5188, 1091]
```

**Post-padding**: Add zeros at end
```
[2423, 5188, 1091] → [2423, 5188, 1091, 0, 0, 0, 0, 0]
```

Use pre-padding for LSTM (recent words more important).

**Q4: Why do we need padding?**

Neural networks need fixed input size:
- Sentences have variable lengths
- Padding makes all sentences same length
- Allows batch processing

Example: 4-word sentence + 4 zeros = 8 (matches max_length)

**Q5: How many parameters does Embedding layer have?**

$$\text{Parameters} = \text{vocab\_size} \times \text{embedding\_dim}$$

Example:
- vocab_size = 500
- embedding_dim = 10
- Parameters = 500 × 10 = 5000

**Q6: What is the difference between Embedding layer and Word2Vec?**

| Aspect | Embedding Layer | Word2Vec |
|--------|----------------|----------|
| Training | Part of model | Pre-trained/separate |
| Task | Task-specific | General purpose |
| Updates | During training | Fixed |
| Use case | Custom datasets | Transfer learning |

**Q7: What is vocabulary size and how to choose it?**

Vocabulary size = number of unique words to consider.

**Choosing:**
- Small dataset: 500-1000
- Medium dataset: 5000-10000
- Large dataset: 20000-50000

Trade-off: Larger = more accurate but slower.

**Q8: What happens if we use post-padding instead of pre-padding in LSTM?**

**Pre-padding (recommended)**:
- Zeros at beginning
- LSTM processes zeros first, then actual words
- Final state has recent word information

**Post-padding**:
- Zeros at end
- LSTM processes actual words, then zeros
- Final state has zero information (not ideal)

**Q9: How does one-hot encoding work in TensorFlow?**

```python
one_hot(sentence, vocab_size)
```

Returns **indices** where 1 exists in sparse vector:
- Input: "I have a glass"
- Output: [2423, 5188, 1091, 5336]

Each number = index in vocabulary where value is 1.

**Q10: What is the output shape of Embedding layer?**

```python
Input: (batch_size, max_length)
Output: (batch_size, max_length, embedding_dim)
```

Example:
- batch_size = 32 sentences
- max_length = 8 words
- embedding_dim = 10
- Output shape: (32, 8, 10)

**Q11: When should you use Embedding layer vs Word2Vec?**

**Use Embedding Layer:**
- Have custom dataset
- Training end-to-end model
- Task-specific embeddings
- Sufficient training data

**Use Word2Vec:**
- Small dataset
- General purpose text
- Transfer learning
- Pre-trained model available (Google News)

**Q12: What is the input to Embedding layer?**

**Input**: Padded sequences of word indices

Example:
```python
[
    [0, 0, 0, 0, 2423, 5188, 1091, 5336],  # Sentence 1
    [0, 0, 0, 142, 234, 91, 178, 321],     # Sentence 2
    ...
]
```

Each number is a word index (from one-hot encoding).

## 💡 Key Takeaways

- **Text Preprocessing Pipeline**: Sentences → One-hot → Padding → Embedding → Vectors
- **One-Hot Encoding**: Converts words to indices (TensorFlow: `one_hot`)
- **Vocabulary Size**: Number of unique words (500-50000 typical)
- **Padding**: Makes all sentences same length (use pre-padding for LSTM)
- **Embedding Layer**: Converts indices to dense vectors (trainable)
- **Parameters**: vocab_size × embedding_dim
- **Embedding vs Word2Vec**: Embedding layer trains with model, Word2Vec pre-trained
- **Output Shape**: (batch, max_length, embedding_dim)
- **Use Case**: Custom datasets, end-to-end training, task-specific embeddings

## ⚠️ Common Mistakes

**Mistake 1**: "Use post-padding for all cases"
- **Reality**: Use pre-padding for LSTM (zeros at beginning better)

**Mistake 2**: "Embedding layer and Word2Vec are same"
- **Reality**: Embedding trains with model, Word2Vec is separate/pre-trained

**Mistake 3**: "Vocabulary size = number of sentences"
- **Reality**: Vocabulary size = number of unique words

**Mistake 4**: "One-hot encoding creates sparse vectors"
- **Reality**: In TensorFlow, returns indices only (not full sparse vector)

**Mistake 5**: "Embedding dimension must match vocab size"
- **Reality**: Independent parameters (vocab_size for input, embedding_dim for output)

**Mistake 6**: "No padding needed if sentences similar length"
- **Reality**: All sentences MUST be exactly same length for batching

## 📝 Quick Revision Points

### Text Preprocessing Steps

1. **Sentences** (raw text)
2. **One-hot encoding** (get indices)
3. **Vocabulary size** (unique words count)
4. **Padding** (make same length)
5. **Embedding layer** (convert to vectors)

### Padding

**Pre-padding** (recommended for LSTM):
```
[2423, 5188] → [0, 0, 0, 0, 0, 0, 2423, 5188]
```

**Post-padding**:
```
[2423, 5188] → [2423, 5188, 0, 0, 0, 0, 0, 0]
```

### Embedding Layer

```python
model.add(Embedding(
    input_dim=vocab_size,      # 500
    output_dim=embedding_dim,  # 10
    input_length=max_length    # 8
))
```

**Parameters**: 500 × 10 = 5000

### Complete Code

```python
# Step 1: One-hot encoding
encoded = [one_hot(s, vocab_size) for s in sentences]

# Step 2: Padding
padded = pad_sequences(encoded, maxlen=max_length, padding='pre')

# Step 3: Embedding layer
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.compile(optimizer='adam', loss='mse')

# Step 4: Get vectors
vectors = model.predict(padded)
```

### Remember

- **One-hot** = Get word indices
- **Padding** = Make same length (pre-padding for LSTM)
- **Embedding layer** = Trainable word vectors
- **Vocab size** = Unique words count
- **Embedding dim** = Vector size (10, 100, 300)
- **Max length** = Sentence length after padding
- **Parameters** = vocab_size × embedding_dim
