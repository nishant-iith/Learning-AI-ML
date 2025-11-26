# Chapter 4: Average Word2Vec and Training Word2Vec

## 🎯 Learning Objectives
- Understand the problem with variable-length sentence vectors
- Master Average Word2Vec concept and calculation
- Learn how to train Word2Vec from scratch
- Know when to use pre-trained vs custom Word2Vec models
- Understand Word2Vec model parameters
- Learn practical applications and model operations

## 📚 Key Concepts

### The Problem with Basic Word2Vec

#### Variable Dimension Issue

**Scenario**: Machine Learning models need **fixed-size** input vectors

**Word2Vec Limitation:**

```
Sentence: "I want to eat pizza"

Using Word2Vec (300 dimensions):
- "I"     → 300 dimensions
- "want"  → 300 dimensions
- "to"    → 300 dimensions
- "eat"   → 300 dimensions
- "pizza" → 300 dimensions

Total: 5 words × 300 = 1500 dimensions
```

**Problem**: Different sentences have different numbers of words!

```
Sentence 1: "I like pizza"         → 3 × 300 = 900 dimensions
Sentence 2: "I want to eat pizza"  → 5 × 300 = 1500 dimensions
Sentence 3: "Food is good"         → 3 × 300 = 900 dimensions
```

**Issue**: ML models require **fixed input size**, but we're getting variable sizes!

#### Visual Representation

```
Word2Vec vectors for each word:
┌─────────────────────────────────────────────────┐
│ "please"    → [0.23, 0.45, ..., 0.12] (300 dims)│
│ "subscribe" → [0.67, 0.89, ..., 0.34] (300 dims)│
│ "krish"     → [0.12, 0.56, ..., 0.78] (300 dims)│
│ "channel"   → [0.89, 0.23, ..., 0.45] (300 dims)│
└─────────────────────────────────────────────────┘
       ↓
Total: 4 words × 300 = 1200 dimensions
       ↓
Need: FIXED 300 dimensions for ML input!
```

### Average Word2Vec (Avg Word2Vec)

#### The Solution

**Concept**: Average all word vectors in a sentence to get a **fixed-size** sentence vector

**Formula:**

$$\text{Avg Word2Vec}(\text{sentence}) = \frac{1}{n} \sum_{i=1}^{n} \text{Word2Vec}(w_i)$$

Where:
- $n$ = Number of words in sentence
- $w_i$ = $i^{th}$ word in sentence
- Word2Vec($w_i$) = Vector representation of word $i$

#### Step-by-Step Calculation

**Example Sentence**: "krish data science channel"

**Assumption**: Word2Vec dimension = 5 (for simplicity)

**Step 1: Get Word2Vec for each word**

```
Word         Dimension 1  Dimension 2  Dimension 3  Dimension 4  Dimension 5
─────────────────────────────────────────────────────────────────────────────
krish        0.12         0.45         0.23         0.67         0.89
data         0.34         0.56         0.78         0.12         0.45
science      0.67         0.23         0.45         0.89         0.12
channel      0.89         0.34         0.12         0.56         0.23
```

**Step 2: Add vectors dimension-wise**

```
Dimension 1: 0.12 + 0.34 + 0.67 + 0.89 = 2.02
Dimension 2: 0.45 + 0.56 + 0.23 + 0.34 = 1.58
Dimension 3: 0.23 + 0.78 + 0.45 + 0.12 = 1.58
Dimension 4: 0.67 + 0.12 + 0.89 + 0.56 = 2.24
Dimension 5: 0.89 + 0.45 + 0.12 + 0.23 = 1.69
```

**Step 3: Calculate average (divide by 4 words)**

```
Average Word2Vec = [2.02/4, 1.58/4, 1.58/4, 2.24/4, 1.69/4]
                 = [0.505, 0.395, 0.395, 0.560, 0.423]
```

**Result**: Fixed 5-dimensional vector for entire sentence!

#### Complete Example (300 dimensions)

**Real-World Scenario:**

```
Sentence: "please subscribe krish channel"

Word2Vec (300 dims each):
┌──────────────────────────────────────────────────┐
│ "please"    → [a₁, a₂, a₃, ..., a₃₀₀]           │
│ "subscribe" → [b₁, b₂, b₃, ..., b₃₀₀]           │
│ "krish"     → [c₁, c₂, c₃, ..., c₃₀₀]           │
│ "channel"   → [d₁, d₂, d₃, ..., d₃₀₀]           │
└──────────────────────────────────────────────────┘
                    ↓ Average Word2Vec
         [(a₁+b₁+c₁+d₁)/4, (a₂+b₂+c₂+d₂)/4, ..., (a₃₀₀+b₃₀₀+c₃₀₀+d₃₀₀)/4]
                    ↓
            [v₁, v₂, v₃, ..., v₃₀₀]
                    ↓
        Fixed 300-dimensional sentence vector
```

#### Key Insights

**1. Fixed Dimension Output:**
- Sentence with 3 words → Average → 300 dimensions
- Sentence with 10 words → Average → 300 dimensions
- Sentence with 100 words → Average → 300 dimensions

**2. Dimension-wise Operation:**
- For each dimension index (1 to 300):
  - Add values from all word vectors at that index
  - Divide by number of words
  - Store average

**3. Preserves Semantic Meaning:**
- Similar sentences have similar averaged vectors
- Semantic information captured across all words

### Training Word2Vec from Scratch

#### When to Train from Scratch?

**Decision Criteria:**

| Scenario | Recommendation |
|----------|----------------|
| **Pre-trained vocabulary covers 75%+ of your text** | Use pre-trained model |
| **Domain-specific vocabulary (medical, legal, finance)** | Train from scratch |
| **New words/slang not in pre-trained model** | Train from scratch |
| **Small dataset (< 10K documents)** | Use pre-trained model |
| **Large dataset (> 100K documents)** | Train from scratch |

**Example Scenarios:**

**Use Pre-trained:**
- General sentiment analysis
- News classification
- Product reviews
- Social media text (general topics)

**Train from Scratch:**
- Medical diagnosis (technical terms)
- Legal document analysis (legal jargon)
- Scientific papers (domain-specific terms)
- Company-specific chatbots (internal terminology)

#### Gensim Library

**Installation:**

```
pip install gensim
```

**Import:**

```python
import gensim
from gensim.models import Word2Vec
```

#### Model Parameters

**Essential Parameters:**

| Parameter | Description | Default | When to Change |
|-----------|-------------|---------|----------------|
| **sentences** | Training data (list of tokenized sentences) | Required | Always provide |
| **vector_size** | Output dimension size | 100 | Use 300 for better quality |
| **window** | Context window size (left + right) | 5 | Larger for more context |
| **min_count** | Ignore words with frequency < this | 5 | Lower (2) to keep rare words |
| **workers** | CPU threads for training | 3 | More for faster training |
| **sg** | Training algorithm (0=CBOW, 1=Skip-gram) | 0 | 1 for small datasets |
| **epochs** | Number of training iterations | 5 | More (10-20) for better training |

**Parameter Details:**

**1. vector_size (Dimensions)**

```
vector_size = 100  → Each word represented by 100 dimensions
vector_size = 300  → Each word represented by 300 dimensions
```

**Rule of Thumb**:
- 100-300 dimensions for most tasks
- Google News model uses 300 dimensions

**2. window (Context Window)**

```
Sentence: "I love to eat pizza"

window = 2:
- For "eat": Context = ["love", "to", "pizza"]  (2 left + 2 right)

window = 5:
- For "eat": Context = ["I", "love", "to", "pizza"]  (all words, limited by sentence)
```

**Rule of Thumb**:
- window = 5 (default): Good for most tasks
- Larger window (7-10): Captures broader context
- Smaller window (2-3): Focuses on immediate neighbors

**3. min_count (Minimum Frequency)**

```
min_count = 2:
- Word "pizza" appears 1 time → Ignored
- Word "food" appears 3 times → Included

min_count = 5:
- Word "good" appears 10 times → Included
- Word "excellent" appears 2 times → Ignored
```

**Rule of Thumb**:
- Large dataset: min_count = 5 (default)
- Small dataset: min_count = 1 or 2
- Remove noise: min_count = 10+

**4. sg (Training Algorithm)**

```
sg = 0 → CBOW (Continuous Bag of Words)
sg = 1 → Skip-gram
```

**When to Use:**
- **CBOW (sg=0)**: Large datasets, faster training
- **Skip-gram (sg=1)**: Small datasets, rare words, better quality

#### Training Process

**Step 1: Prepare Data**

```
Input: Raw sentences
['I love pizza', 'Food is good', 'Pizza tastes amazing']

After preprocessing:
[['love', 'pizza'], ['food', 'good'], ['pizza', 'taste', 'amazing']]
```

**Step 2: Train Model**

```python
model = Word2Vec(
    sentences=preprocessed_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)
```

**Step 3: Model is Ready**

```
Trained model with:
- Vocabulary size: 5000 words
- Vector dimension: 100
- Training epochs: 10
```

### Word2Vec Model Operations

#### 1. Access Vocabulary

**Get all vocabulary words:**

```python
vocabulary = model.wv.index_to_key
```

**Example Output:**
```
['food', 'good', 'pizza', 'love', 'eat', 'delicious', ...]
```

**Check vocabulary size:**

```python
vocab_size = len(model.wv)
```

**Example Output:**
```
Vocabulary size: 5564 words
```

#### 2. Get Word Vector

**Extract vector for specific word:**

```python
vector = model.wv['pizza']
```

**Example Output:**
```
array([0.23, 0.45, 0.67, ..., 0.89], dtype=float32)  # 100 dimensions
```

**Vector properties:**
- Length = vector_size (e.g., 100)
- Each value is a float between -1 and 1 (typically)

#### 3. Find Similar Words

**Get most similar words:**

```python
similar = model.wv.most_similar('pizza', topn=5)
```

**Example Output:**
```
[('food', 0.87),
 ('pasta', 0.82),
 ('burger', 0.79),
 ('delicious', 0.76),
 ('restaurant', 0.73)]
```

**Interpretation**:
- First value: Similar word
- Second value: Cosine similarity (0-1, higher = more similar)

#### 4. Calculate Similarity

**Similarity between two words:**

```python
similarity = model.wv.similarity('pizza', 'food')
```

**Example Output:**
```
0.87  (87% similar)
```

**Interpretation**:
- 0.0 = Not similar at all
- 0.5 = Moderately similar
- 1.0 = Identical (same word)

#### 5. Word Arithmetic

**Famous example: king - man + woman = queen**

```python
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
```

**Example Output:**
```
[('queen', 0.82)]
```

**More Examples:**

```
France - Paris + London = England
Apple - iOS + Android = Samsung
Doctor - Hospital + School = Teacher
```

### Pre-trained Word2Vec Models

#### Google News Word2Vec

**Specifications:**
- **Training Data**: Google News articles (3 billion words)
- **Vocabulary**: 3 million words
- **Dimensions**: 300
- **File Size**: ~1.5 GB

**When to Use:**
- General English text
- News articles
- Social media
- Product reviews
- Any text with common vocabulary

**Loading Pre-trained Model:**

```python
from gensim.models import KeyedVectors

# Load Google News Word2Vec
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True
)
```

**Usage:**

```python
# Get similar words
similar = model.most_similar('computer')

Output:
[('computers', 0.74),
 ('laptop', 0.69),
 ('PC', 0.65),
 ('software', 0.63)]
```

#### Advantages of Pre-trained Models

✓ **No Training Required**: Ready to use immediately
✓ **Large Vocabulary**: Covers 3 million words
✓ **High Quality**: Trained on billions of words
✓ **General Purpose**: Works for most English text
✓ **Saves Time**: No need for large training dataset

#### Disadvantages of Pre-trained Models

✗ **Domain-Specific Words Missing**: Medical/legal terms may not be present
✗ **Large File Size**: ~1.5 GB download
✗ **Fixed Dimensions**: Cannot change to 100 or 200
✗ **No Custom Training**: Cannot adapt to your specific vocabulary

## Practical Application: Text Classification

### Complete Workflow

**Step 1: Text Preprocessing**

```
Raw Sentences:
- "The food is very good"
- "Bad service"
- "Amazing pizza!"

After Preprocessing:
- ['food', 'good']
- ['bad', 'service']
- ['amazing', 'pizza']
```

**Step 2: Train Word2Vec or Load Pre-trained**

```
Option A: Train from scratch
model = Word2Vec(sentences, vector_size=100, window=5)

Option B: Use pre-trained
model = KeyedVectors.load_word2vec_format('GoogleNews...')
```

**Step 3: Convert Sentences to Vectors using Average Word2Vec**

```
Sentence 1: ['food', 'good']
- Get vector for 'food': [0.12, 0.45, ..., 0.89]
- Get vector for 'good': [0.34, 0.67, ..., 0.23]
- Average: [(0.12+0.34)/2, (0.45+0.67)/2, ..., (0.89+0.23)/2]
- Result: [0.23, 0.56, ..., 0.56]  → Fixed 100 dims

Sentence 2: ['bad', 'service']
- Average vectors
- Result: [0.45, 0.78, ..., 0.34]  → Fixed 100 dims
```

**Step 4: Create Input Matrix**

```
Input Features (X):
┌───────────────────────────────────┐
│ [0.23, 0.56, ..., 0.56]  ← Sent 1│
│ [0.45, 0.78, ..., 0.34]  ← Sent 2│
│ [0.67, 0.23, ..., 0.89]  ← Sent 3│
└───────────────────────────────────┘
Shape: (3 sentences, 100 dimensions)

Output Labels (y):
[1, 0, 1]  (1=Positive, 0=Negative)
```

**Step 5: Train ML Model**

```
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
```

### Example: Sentiment Analysis

**Dataset**: Movie reviews

**Preprocessing:**

```
Review 1: "This movie is amazing and wonderful"
→ Cleaned: ['movie', 'amazing', 'wonderful']
→ Label: Positive (1)

Review 2: "Terrible acting and boring plot"
→ Cleaned: ['terrible', 'acting', 'boring', 'plot']
→ Label: Negative (0)
```

**Word2Vec Training:**

```python
# Collect all preprocessed reviews
all_reviews = [
    ['movie', 'amazing', 'wonderful'],
    ['terrible', 'acting', 'boring', 'plot'],
    ...  # thousands more
]

# Train Word2Vec
model = Word2Vec(
    sentences=all_reviews,
    vector_size=300,
    window=5,
    min_count=2,
    epochs=10
)
```

**Average Word2Vec Function:**

```python
def average_word2vec(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)
```

**Convert All Reviews:**

```python
X = []
for review in all_reviews:
    avg_vector = average_word2vec(review, model)
    X.append(avg_vector)

X = np.array(X)  # Shape: (num_reviews, 300)
```

**Train Classifier:**

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Advantages and Disadvantages

### Advantages of Average Word2Vec

**1. Fixed Dimension Output**
- Same vector size regardless of sentence length
- Compatible with ML models

**2. Semantic Meaning Preserved**
- Averages capture overall sentence meaning
- Similar sentences have similar vectors

**3. Simple to Implement**
- One line: `np.mean(vectors, axis=0)`
- Fast computation

**4. Works with Pre-trained Models**
- Can use Google News Word2Vec
- No need to train from scratch

**5. Better than Count-based Methods**
- Captures semantic relationships
- Dense vectors (no sparsity)

### Disadvantages of Average Word2Vec

**1. Word Order Lost**
- "dog bites man" = "man bites dog" (same average)
- Cannot capture sequence information

**2. Equal Weight to All Words**
- Important words (not, very) treated same as others
- No attention mechanism

**3. Context Not Preserved**
- Loses sentence structure
- "good" and "not good" may have similar averages

**4. Rare Word Handling**
- Out-of-vocabulary words ignored
- Affects sentence representation

**5. Not Ideal for Long Texts**
- Averaging 100+ word vectors dilutes meaning
- Better suited for short sentences

## ❓ Interview Questions & Answers

**Q1: What is Average Word2Vec and why is it needed?**

Average Word2Vec averages all word vectors in a sentence to create a fixed-size sentence vector. Needed because:
- Word2Vec gives each word a vector (e.g., 300 dims)
- A sentence with N words would have N × 300 dimensions (variable size)
- ML models need fixed input size
- Average Word2Vec solves this: Any sentence → Fixed 300 dimensions

**Q2: How do you calculate Average Word2Vec?**

Formula: $\text{Avg}(\text{sentence}) = \frac{1}{n} \sum_{i=1}^{n} \text{Word2Vec}(w_i)$

Steps:
1. Get Word2Vec vector for each word
2. Add all vectors dimension-wise
3. Divide by number of words
4. Result: Fixed-size sentence vector

**Q3: When should you train Word2Vec from scratch vs use pre-trained models?**

**Train from Scratch:**
- Domain-specific vocabulary (medical, legal)
- Pre-trained model covers < 75% of your vocabulary
- Large custom dataset available (> 100K documents)

**Use Pre-trained:**
- General text (news, reviews, social media)
- Pre-trained covers > 75% vocabulary
- Small dataset (< 10K documents)

**Q4: What are the key parameters in training Word2Vec?**

- **vector_size**: Output dimension (default: 100, use 300 for quality)
- **window**: Context window size (default: 5)
- **min_count**: Minimum word frequency (default: 5, lower to keep rare words)
- **sg**: Training algorithm (0=CBOW, 1=Skip-gram)
- **epochs**: Training iterations (default: 5, more for better quality)

**Q5: What is the difference between CBOW and Skip-gram?**

- **CBOW (sg=0)**: Context words → Target word. Faster, good for large datasets.
- **Skip-gram (sg=1)**: Target word → Context words. Better for rare words and small datasets.

**Q6: How does the window parameter work in Word2Vec?**

Window = number of context words on each side.

Example: "I love to eat pizza" with window=2
- For "eat": Context = ["love", "to", "pizza"]  (2 left, 2 right)

Larger window = more context, broader relationships.

**Q7: What is the min_count parameter?**

min_count ignores words with frequency below threshold.

Example: min_count=2
- Word "pizza" appears 1 time → Ignored
- Word "food" appears 3 times → Included

Purpose: Remove rare words, reduce vocabulary size, filter noise.

**Q8: How do you find similar words using Word2Vec?**

```python
similar = model.wv.most_similar('pizza', topn=5)
```

Returns: List of (word, similarity_score) tuples
- Similarity score: 0 to 1 (higher = more similar)
- Based on cosine similarity of word vectors

**Q9: What is word arithmetic in Word2Vec? Give example.**

Word arithmetic: Algebraic operations on word vectors to find analogies.

Famous example: king - man + woman = queen

```python
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man']
)
# Output: 'queen'
```

Other examples:
- France - Paris + London = England
- Apple - iOS + Android = Samsung

**Q10: What are the advantages of Average Word2Vec over Bag of Words?**

| Aspect | Bag of Words | Average Word2Vec |
|--------|--------------|------------------|
| **Sparsity** | Sparse (mostly zeros) | Dense (all values filled) |
| **Semantic Meaning** | Not captured | Captured |
| **Dimension** | Vocabulary size (10K+) | Fixed (100-300) |
| **Similar Words** | Different vectors | Similar vectors |

Average Word2Vec is better for capturing meaning and reducing dimensions.

**Q11: What are the disadvantages of Average Word2Vec?**

1. **Word Order Lost**: "dog bites man" = "man bites dog"
2. **Equal Weight**: All words weighted equally (no attention)
3. **Context Lost**: "good" and "not good" may average similarly
4. **OOV Words**: Out-of-vocabulary words ignored

**Q12: What is Google News Word2Vec model?**

Pre-trained Word2Vec model:
- **Training Data**: 3 billion words from Google News
- **Vocabulary**: 3 million words
- **Dimensions**: 300
- **File Size**: ~1.5 GB

Use for general English text without training from scratch.

## 💡 Key Takeaways

- **Average Word2Vec** = Average all word vectors to get fixed-size sentence vector
- **Formula**: $\text{Avg} = \frac{1}{n} \sum_{i=1}^{n} \text{Word2Vec}(w_i)$
- **Solves Problem**: Variable sentence length → Fixed vector size
- **Training from Scratch**: Use Gensim library with proper parameters
- **vector_size**: Output dimension (100-300 typical)
- **window**: Context window size (5 default)
- **min_count**: Minimum word frequency (2-5 typical)
- **sg**: 0=CBOW (fast), 1=Skip-gram (quality)
- **Pre-trained vs Custom**: Use pre-trained if 75%+ vocabulary coverage
- **Word Arithmetic**: king - man + woman = queen
- **Similarity**: Cosine similarity measures word relationship (0-1)
- **Application**: Text classification with fixed-size inputs

## ⚠️ Common Mistakes

**Mistake 1**: "Average Word2Vec captures word order"
- **Reality**: Averaging loses word order ("dog bites man" = "man bites dog")

**Mistake 2**: "Use Average Word2Vec for all tasks"
- **Reality**: Not ideal for tasks needing word order (use RNN/LSTM/Transformers)

**Mistake 3**: "Always train Word2Vec from scratch"
- **Reality**: Use pre-trained if vocabulary coverage > 75%

**Mistake 4**: "Larger window is always better"
- **Reality**: Larger window captures broader context but may dilute meaning

**Mistake 5**: "Average Word2Vec handles 'not good' correctly"
- **Reality**: "good" and "not good" may have similar averages (negation lost)

**Mistake 6**: "Set min_count=1 for all datasets"
- **Reality**: Low min_count includes noisy/rare words; use 2-5 for cleaner vocabulary

## 📝 Quick Revision Points

### Average Word2Vec

**Problem:**
```
Sentence: 5 words × 300 dims = 1500 dims (variable!)
```

**Solution:**
```
Average Word2Vec: Any sentence → 300 dims (fixed!)
```

**Formula:**

$$\text{Avg Word2Vec} = \frac{1}{n} \sum_{i=1}^{n} \text{Word2Vec}(w_i)$$

### Training Word2Vec

**Gensim Syntax:**

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=data,
    vector_size=300,    # Output dimension
    window=5,           # Context window
    min_count=2,        # Minimum frequency
    sg=0,               # 0=CBOW, 1=Skip-gram
    epochs=10           # Training iterations
)
```

### Key Parameters

| Parameter | Default | Typical Range | Purpose |
|-----------|---------|---------------|---------|
| **vector_size** | 100 | 100-300 | Output dimension size |
| **window** | 5 | 2-10 | Context window size |
| **min_count** | 5 | 1-10 | Ignore rare words |
| **sg** | 0 | 0 or 1 | CBOW(0) or Skip-gram(1) |
| **epochs** | 5 | 5-20 | Training iterations |

### Model Operations

**Get Similar Words:**
```python
model.wv.most_similar('pizza', topn=5)
# Output: [('food', 0.87), ('pasta', 0.82), ...]
```

**Get Vector:**
```python
vector = model.wv['pizza']  # Shape: (300,)
```

**Calculate Similarity:**
```python
similarity = model.wv.similarity('pizza', 'food')  # 0.87
```

**Word Arithmetic:**
```python
model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
# Output: [('queen', 0.82)]
```

### Pre-trained vs Custom

**Pre-trained (Google News):**
- 3 billion words
- 3 million vocabulary
- 300 dimensions
- 1.5 GB file

**When to Use:**
- Vocabulary coverage > 75%
- General English text
- Small dataset

**Train from Scratch:**
- Domain-specific vocabulary
- Large custom dataset
- Unique terminology

### Remember

- **Average Word2Vec** = Fixed-size sentence vector
- **Dimension-wise averaging** = Sum each dimension, divide by word count
- **Word order lost** = Averaging cannot preserve sequence
- **Pre-trained models** = Use for general text to save time
- **vector_size** = Higher (300) is better quality
- **window** = Larger captures more context
- **min_count** = Higher filters noise
- **CBOW** = Fast, large datasets
- **Skip-gram** = Quality, small datasets
