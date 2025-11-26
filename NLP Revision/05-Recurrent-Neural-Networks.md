# Chapter 5: Recurrent Neural Networks (RNN)

## 🎯 Learning Objectives
- Understand why RNN is needed for sequence data
- Learn RNN architecture and feedback mechanism
- Master types of RNN (one-to-one, one-to-many, many-to-one, many-to-many)
- Understand forward propagation in RNN
- Know RNN applications and use cases
- Learn when to use RNN vs traditional ML

## 📚 Key Concepts

### Why RNN?

#### Limitations of Traditional Methods

**Problem with ML/BoW/TF-IDF/Avg Word2Vec:**

```
Sentence 1: "dog bites man"
Sentence 2: "man bites dog"

Using Average Word2Vec:
- Both sentences have SAME vectors (order lost!)
- But meanings are COMPLETELY different!
```

**Key Issue**: **Word order** and **sequence information** is lost

#### Sequence-Dependent Applications

**Applications where sequence matters:**

**1. Chatbot / Question Answering**

```
Question: "What is the weather like outside?"

- "What" + "is" + "the" + "weather" → Context builds up
- "like" + "outside" → Completes meaning
- Sequence = Grammatically correct, meaningful question
```

**If word order changes:**
```
"Outside weather the is what like?" → Nonsense!
```

**2. Language Translation**

```
English: "I love to eat pizza"
Hindi:   "मुझे पिज्जा खाना पसंद है"

- Word order different in different languages
- Grammatical structure must be preserved
- Sequential translation needed
```

**3. Text Generation / Auto-Completion**

```
You type: "The food is"

Auto-suggestions:
- "The food is good"
- "The food is delicious"
- "The food is amazing"

→ Model predicts NEXT word based on SEQUENCE
```

**4. Sentiment Analysis**

```
Sentence 1: "The food is good"          → Positive
Sentence 2: "The food is not good"      → Negative

"not" changes entire meaning!
→ Sequence and context critical
```

#### Why Machine Learning Fails

**Machine Learning (BoW, TF-IDF, Avg Word2Vec):**

```
Input:  Sentence → Vector (fixed size)
Output: Prediction

Problem:
- NO memory of previous words
- NO sequential processing
- Word order LOST
```

**Deep Learning (RNN):**

```
Input:  Word₁ → Word₂ → Word₃ → Word₄ → ...
Processing: Each word processed sequentially
Memory: Previous words remembered
Output: Context-aware prediction
```

**Key Difference**: RNN has **memory** and processes **sequentially**

### RNN Applications

**1. Chatbots**
- User asks question word-by-word
- Bot understands context from sequence
- Generates grammatically correct response

**2. Language Translation**
- English sentence → Sequential processing → Hindi output
- Word order preserved
- Grammar maintained

**3. Text Generation**
- Start with seed text: "Once upon a"
- Generate next word: "time"
- Continue: "there was a"
- Generate: "king"
- Result: "Once upon a time there was a king"

**4. Sentiment Analysis**
- Process "The food is not good" sequentially
- "not" negates "good"
- Final sentiment: Negative

**5. Speech Recognition**
- Audio → Sequential phonemes → Words → Sentences
- Order critical for understanding

**6. Music Generation**
- Generate notes sequentially
- Maintain musical structure
- Create melodies

**7. Time Series Forecasting**
- Stock prices: Previous values → Predict next
- Weather: Historical data → Future prediction
- Sales: Past trends → Future sales

### Basic RNN Architecture

#### Feedback Loop Concept

**Traditional Neural Network:**

```
Input → [Hidden Layer] → Output
```

**Recurrent Neural Network:**

```
Input → [Hidden Layer] ⟲ → Output
           ↑__________|
         (Feedback)
```

**Key Feature**: Output is fed back to the same network

#### Basic RNN Diagram

```
     ┌─────────┐
     │  RNN    │
x₁ →│  Cell   │→ output₁
     │         │
     └────┬────┘
          │ (feedback)
          ↓
     ┌─────────┐
     │  RNN    │
x₂ →│  Cell   │→ output₂
     │         │
     └────┬────┘
          │ (feedback)
          ↓
     ┌─────────┐
     │  RNN    │
x₃ →│  Cell   │→ output₃
     │         │
     └─────────┘
```

**Explanation:**
- Same RNN cell used repeatedly
- Each time step receives:
  - New input (xₜ)
  - Previous output (outputₜ₋₁)
- Output at time t depends on current input AND previous state

#### Unfolded RNN Architecture

**Folded View (Compact):**

```
    ┌────────┐
x →│  RNN   │→ h
    │  ⟲    │
    └────────┘
```

**Unfolded View (Time Steps):**

```
    ┌────┐      ┌────┐      ┌────┐      ┌────┐
x₁ →│RNN │→ h₁ →│RNN │→ h₂ →│RNN │→ h₃ →│RNN │→ h₄
    └────┘      └────┘      └────┘      └────┘
     t=1         t=2         t=3         t=4
```

**Key Points:**
- **Same weights** used at all time steps
- **Sequential processing**: t=1, then t=2, then t=3, ...
- **Hidden state** (h) carries information forward

### Types of RNN

#### 1. One-to-One RNN

**Structure:**

```
    ┌────┐
x →│RNN │→ y
    └────┘

Input:  1 element
Output: 1 element
```

**Example: Image Classification**

```
Input:  Image (single input)
Output: Class label (single output)

Example:
Image of cat → [RNN] → "Cat"
```

**Note**: Not common for RNN (CNNs better for images)

#### 2. One-to-Many RNN

**Structure:**

```
    ┌────┐   ┌────┐   ┌────┐   ┌────┐
x →│RNN │→ │RNN │→ │RNN │→ │RNN │
    └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
      ↓        ↓        ↓        ↓
      y₁       y₂       y₃       y₄

Input:  1 element
Output: Multiple elements (sequence)
```

**Example 1: Music Generation**

```
Input:  Starting note (C)
Output: Sequence of notes (C → D → E → F → G)

Process:
1. Input "C" → Generate "D"
2. Use "D" → Generate "E"
3. Use "E" → Generate "F"
4. Continue...
```

**Example 2: Image Captioning**

```
Input:  Image (single input)
Output: Caption (sequence of words)

Image of dog playing → [RNN] → "A" "dog" "is" "playing" "in" "park"
```

**Example 3: Text Generation**

```
Input:  Seed word ("Once")
Output: Generated sentence ("Once upon a time there was a king")
```

#### 3. Many-to-One RNN

**Structure:**

```
    ┌────┐   ┌────┐   ┌────┐   ┌────┐
x₁ →│RNN │→ │RNN │→ │RNN │→ │RNN │
    └────┘   └────┘   └────┘   └─┬──┘
                                  ↓
                                  y

Input:  Multiple elements (sequence)
Output: 1 element
```

**Example 1: Sentiment Analysis**

```
Input:  "The" "food" "is" "very" "good"
         ↓     ↓      ↓     ↓      ↓
       [RNN]→[RNN]→[RNN]→[RNN]→[RNN]
                                  ↓
                             "Positive"

Process:
1. Read "The" → Update hidden state
2. Read "food" → Update hidden state
3. Read "is" → Update hidden state
4. Read "very" → Update hidden state
5. Read "good" → Final prediction: Positive
```

**Example 2: Document Classification**

```
Input:  Document (sequence of words)
Output: Category (Sports, Politics, Entertainment)
```

**Example 3: Next Day Sales Prediction**

```
Input:  Sales history (Day 1, Day 2, ..., Day 30)
Output: Predicted sales for Day 31
```

#### 4. Many-to-Many RNN

**Structure (Same Length):**

```
    ┌────┐   ┌────┐   ┌────┐   ┌────┐
x₁ →│RNN │→ │RNN │→ │RNN │→ │RNN │
    └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
      ↓        ↓        ↓        ↓
      y₁       y₂       y₃       y₄

Input:  Multiple elements
Output: Multiple elements (same length)
```

**Example 1: Named Entity Recognition (NER)**

```
Input:  "John"  "lives" "in"   "Paris"
         ↓       ↓       ↓       ↓
       [RNN]   [RNN]   [RNN]   [RNN]
         ↓       ↓       ↓       ↓
Output: "PERSON" "O"    "O"    "LOCATION"
```

**Example 2: Video Classification (Frame-by-Frame)**

```
Input:  Frame₁ → Frame₂ → Frame₃ → Frame₄
Output: Label₁ → Label₂ → Label₃ → Label₄
```

**Structure (Different Length - Encoder-Decoder):**

```
Encoder:                    Decoder:
x₁ → [RNN] → [RNN] → [RNN] → [RNN] → [RNN] → y₁
x₂ ───────↗          ↓         ↓
x₃ ──────────────────↗         ↓
                               y₂
                               y₃
```

**Example: Language Translation**

```
Input:  "I" "love" "pizza" (English - 3 words)
         ↓    ↓      ↓
       [Encoder RNN processes all]
                ↓
       [Decoder RNN generates output]
                ↓
Output: "मुझे" "पिज्जा" "पसंद" "है" (Hindi - 4 words)
```

**Example: Question Answering**

```
Input:  Question (variable length)
         ↓
       [Encoder processes question]
         ↓
       [Decoder generates answer]
         ↓
Output: Answer (variable length)
```

### Forward Propagation in RNN

#### Notation

**Variables:**
- $x_t$ = Input at time step t
- $h_t$ = Hidden state at time step t
- $y_t$ = Output at time step t
- $W$ = Weight matrix for input
- $W_h$ = Weight matrix for hidden state
- $b$ = Bias term

#### Example: Sentiment Analysis (Many-to-One)

**Sentence**: "The food is very good"

**Step 1: Tokenization**

```
Words: ["The", "food", "is", "very", "good"]
Mapped: [x₁, x₂, x₃, x₄, x₅]
```

**Step 2: Convert to Vectors**

Using Word2Vec (assume 300 dimensions):
```
x₁ = Word2Vec("The")   → [0.12, 0.45, ..., 0.89]  (300 dims)
x₂ = Word2Vec("food")  → [0.34, 0.67, ..., 0.23]  (300 dims)
x₃ = Word2Vec("is")    → [0.56, 0.89, ..., 0.45]  (300 dims)
x₄ = Word2Vec("very")  → [0.78, 0.12, ..., 0.67]  (300 dims)
x₅ = Word2Vec("good")  → [0.23, 0.56, ..., 0.12]  (300 dims)
```

**Step 3: RNN Architecture**

```
    ┌────┐      ┌────┐      ┌────┐      ┌────┐      ┌────┐
x₁ →│RNN │→ h₁ →│RNN │→ h₂ →│RNN │→ h₃ →│RNN │→ h₄ →│RNN │→ h₅
    └────┘      └────┘      └────┘      └────┘      └─┬──┘
    t=1         t=2         t=3         t=4          t=5
                                                       ↓
                                                    [Sigmoid/Softmax]
                                                       ↓
                                                   ŷ (Positive/Negative)
```

#### Forward Propagation Equations

**Time Step t=1:**

$$h_1 = f(W \cdot x_1 + b)$$

Where:
- $W$ = Weight matrix for input (initialized randomly)
- $x_1$ = Input vector (300 dims)
- $b$ = Bias
- $f$ = Activation function (tanh or ReLU)

**Time Step t=2:**

$$h_2 = f(W \cdot x_2 + W_h \cdot h_1 + b)$$

Where:
- $W \cdot x_2$ = Current input contribution
- $W_h \cdot h_1$ = Previous hidden state contribution
- $W_h$ = Weight matrix for hidden state

**Time Step t=3:**

$$h_3 = f(W \cdot x_3 + W_h \cdot h_2 + b)$$

**General Formula (Time Step t):**

$$h_t = f(W \cdot x_t + W_h \cdot h_{t-1} + b)$$

**Final Output (Many-to-One):**

$$\hat{y} = \sigma(W_y \cdot h_5 + b_y)$$

Where:
- $\sigma$ = Sigmoid (binary) or Softmax (multi-class)
- $W_y$ = Output weight matrix
- $h_5$ = Final hidden state
- $b_y$ = Output bias

#### Detailed Example with Dimensions

**Setup:**
- Input dimension: 300 (Word2Vec)
- Hidden dimension: 128 (neurons in RNN cell)
- Output: Binary (Positive=1, Negative=0)

**Weight Matrices:**

$$W: (128 \times 300) \text{ - Input to hidden}$$

$$W_h: (128 \times 128) \text{ - Hidden to hidden}$$

$$W_y: (1 \times 128) \text{ - Hidden to output}$$

**Time Step 1:**

$$h_1 = \tanh(W \cdot x_1 + b)$$

$$h_1 = \tanh([128 \times 300] \cdot [300 \times 1] + [128 \times 1])$$

$$h_1: [128 \times 1] \text{ (hidden state vector)}$$

**Time Step 2:**

$$h_2 = \tanh(W \cdot x_2 + W_h \cdot h_1 + b)$$

$$h_2 = \tanh([128 \times 300] \cdot [300 \times 1] + [128 \times 128] \cdot [128 \times 1] + [128 \times 1])$$

$$h_2: [128 \times 1]$$

**Time Step 5 (Final):**

$$h_5 = \tanh(W \cdot x_5 + W_h \cdot h_4 + b)$$

**Output:**

$$\hat{y} = \sigma(W_y \cdot h_5 + b_y)$$

$$\hat{y} = \sigma([1 \times 128] \cdot [128 \times 1] + [1 \times 1])$$

$$\hat{y}: \text{Scalar value between 0 and 1}$$

**Interpretation:**
- $\hat{y} \geq 0.5$ → Positive sentiment
- $\hat{y} < 0.5$ → Negative sentiment

#### Complete Forward Propagation Flow

```
Input: "The food is very good"
       ↓
   [Word2Vec]
       ↓
   x₁, x₂, x₃, x₄, x₅ (each 300 dims)
       ↓
┌──────────────────────────────┐
│ Time Step 1:                 │
│ h₁ = tanh(W·x₁ + b)         │
│ → Hidden state: 128 dims     │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Time Step 2:                 │
│ h₂ = tanh(W·x₂ + Wₕ·h₁ + b)│
│ → Hidden state: 128 dims     │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Time Step 3:                 │
│ h₃ = tanh(W·x₃ + Wₕ·h₂ + b)│
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Time Step 4:                 │
│ h₄ = tanh(W·x₄ + Wₕ·h₃ + b)│
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Time Step 5:                 │
│ h₅ = tanh(W·x₅ + Wₕ·h₄ + b)│
└──────────────────────────────┘
       ↓
   [Output Layer]
   ŷ = σ(Wᵧ·h₅ + bᵧ)
       ↓
   0.87 → Positive!
```

### Backward Propagation in RNN

**Process:**

1. **Calculate Loss:**

$$L = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$$

2. **Compute Gradients:**

$$\frac{\partial L}{\partial W_y}, \frac{\partial L}{\partial W_h}, \frac{\partial L}{\partial W}$$

3. **Backpropagate Through Time (BPTT):**
- Gradient flows backward: t=5 → t=4 → t=3 → t=2 → t=1
- Update weights at each time step

4. **Update Weights:**

$$W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}$$

Where $\alpha$ = Learning rate

**Key Issue**: Vanishing/Exploding Gradients (covered in LSTM chapter)

### Key Features of RNN

#### 1. Shared Weights

**Same weights** used across all time steps:
- $W$ (input weights)
- $W_h$ (hidden weights)
- $W_y$ (output weights)

**Advantage**: Fewer parameters, generalizes better

#### 2. Sequential Processing

**Order matters**:
```
"dog bites man" ≠ "man bites dog"

RNN processes:
t=1: "dog"  → h₁
t=2: "bites" → h₂ (remembers "dog")
t=3: "man"   → h₃ (remembers "dog bites")
```

#### 3. Memory (Hidden State)

**Hidden state carries information:**

$$h_t = f(x_t, h_{t-1})$$

- $h_t$ depends on current input AND previous state
- Information propagates forward through time

#### 4. Variable Length Input/Output

**Flexible architectures:**
- One-to-Many: 1 input → N outputs
- Many-to-One: N inputs → 1 output
- Many-to-Many: N inputs → M outputs (N ≠ M possible)

## Advantages and Disadvantages

### Advantages

**1. Sequential Processing**
- Captures word order and context
- Understands sentence structure

**2. Memory Capability**
- Remembers previous inputs
- Hidden state carries information forward

**3. Variable Length Handling**
- Works with different input sizes
- No need for fixed-size padding (in theory)

**4. Shared Weights**
- Same weights across time steps
- Fewer parameters than separate networks

**5. Temporal Dependencies**
- Captures relationships over time
- Ideal for time series, text, speech

### Disadvantages

**1. Vanishing Gradient Problem**
- Gradients become very small over long sequences
- Difficult to learn long-term dependencies
- Solution: LSTM, GRU

**2. Exploding Gradient Problem**
- Gradients become very large
- Training becomes unstable
- Solution: Gradient clipping

**3. Sequential Processing (Slow)**
- Cannot parallelize across time steps
- Must process t=1, then t=2, then t=3...
- Slow for long sequences

**4. Short-Term Memory**
- Forgets information from early time steps
- Struggles with long sequences (> 100 tokens)
- Solution: LSTM, GRU, Attention

**5. Computational Cost**
- More expensive than feedforward networks
- Requires backpropagation through time (BPTT)

## ❓ Interview Questions & Answers

**Q1: What is RNN and why is it needed?**

RNN (Recurrent Neural Network) is a neural network that processes sequential data by maintaining a hidden state that carries information across time steps. Needed because:
- Traditional ML loses word order ("dog bites man" = "man bites dog")
- Sequence matters in text, speech, time series
- RNN has memory and processes sequentially

**Q2: What is the key difference between RNN and feedforward neural networks?**

**Feedforward Network:**
- Processes entire input at once
- No memory of previous inputs
- Fixed input size

**RNN:**
- Processes input sequentially (one element at a time)
- Maintains hidden state (memory)
- Variable input size
- Feedback loop (output fed back to network)

**Q3: What are the types of RNN architectures?**

1. **One-to-One**: 1 input → 1 output (Image classification)
2. **One-to-Many**: 1 input → N outputs (Music generation, image captioning)
3. **Many-to-One**: N inputs → 1 output (Sentiment analysis, document classification)
4. **Many-to-Many**: N inputs → M outputs (Language translation, NER)

**Q4: What is the forward propagation equation in RNN?**

$$h_t = f(W \cdot x_t + W_h \cdot h_{t-1} + b)$$

Where:
- $h_t$ = Hidden state at time t
- $x_t$ = Input at time t
- $h_{t-1}$ = Previous hidden state
- $W$ = Input weight matrix
- $W_h$ = Hidden state weight matrix
- $f$ = Activation function (tanh or ReLU)

**Q5: Why does RNN have shared weights?**

Same weights ($W$, $W_h$, $W_y$) used across all time steps because:
- Reduces number of parameters
- Generalizes better (learns patterns, not specific positions)
- Makes network scalable to any sequence length

**Q6: What is the hidden state in RNN?**

Hidden state ($h_t$) is the memory of the RNN. It:
- Carries information from previous time steps
- Updated at each time step
- Combines current input with previous state
- Propagates information forward through time

**Q7: Give examples of Many-to-One RNN applications.**

1. **Sentiment Analysis**: Sentence (many words) → Sentiment (positive/negative)
2. **Document Classification**: Document (many words) → Category (sports/politics)
3. **Time Series Prediction**: Historical prices (many) → Next day price (one)
4. **Video Classification**: Video frames (many) → Category (one)

**Q8: Give examples of One-to-Many RNN applications.**

1. **Music Generation**: Starting note → Sequence of notes
2. **Image Captioning**: Image → Caption (sequence of words)
3. **Text Generation**: Seed word → Generated paragraph
4. **Auto-completion**: Partial text → Completed sentence

**Q9: What is the vanishing gradient problem in RNN?**

During backpropagation through time (BPTT):
- Gradients multiplied repeatedly over many time steps
- Gradients become exponentially small
- Early time steps get almost zero gradient
- Network cannot learn long-term dependencies

**Solution**: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)

**Q10: What is backpropagation through time (BPTT)?**

Backpropagation through time is the process of computing gradients in RNN:
1. Unfold RNN across all time steps
2. Calculate loss at output
3. Backpropagate gradients: t=T → t=T-1 → ... → t=1
4. Update shared weights using accumulated gradients

**Q11: Why can't we parallelize RNN training?**

Because RNN processes sequentially:
- t=1 must complete before t=2 can start
- t=2 needs hidden state from t=1
- Cannot compute all time steps simultaneously
- This makes RNN slower than Transformers (which can parallelize)

**Q12: When should you use RNN vs Average Word2Vec?**

**Use Average Word2Vec:**
- Word order not critical
- Short sentences
- Fast inference needed
- Simple classification tasks

**Use RNN:**
- Word order critical ("not good" vs "good")
- Long sequences
- Context matters
- Complex tasks (translation, generation)

## 💡 Key Takeaways

- **RNN** = Recurrent Neural Network with feedback loop
- **Purpose**: Process sequential data (text, speech, time series)
- **Key Feature**: Memory (hidden state) carries information forward
- **Feedback Loop**: Output fed back to same network
- **Sequential Processing**: t=1 → t=2 → t=3 → ... → t=T
- **Shared Weights**: Same W, W_h, W_y across all time steps
- **Types**: One-to-One, One-to-Many, Many-to-One, Many-to-Many
- **Forward Propagation**: $h_t = f(W \cdot x_t + W_h \cdot h_{t-1} + b)$
- **Applications**: Chatbots, translation, sentiment analysis, text generation
- **Advantages**: Sequence processing, memory, variable length
- **Disadvantages**: Vanishing gradients, slow training, short-term memory

## ⚠️ Common Mistakes

**Mistake 1**: "RNN output at time t only depends on input at time t"
- **Reality**: Output at t depends on current input AND all previous inputs (through hidden state)

**Mistake 2**: "RNN has different weights at each time step"
- **Reality**: Same weights shared across ALL time steps

**Mistake 3**: "RNN can handle infinite sequence length"
- **Reality**: Vanishing gradients limit effective memory to ~10-20 time steps

**Mistake 4**: "Use RNN for all NLP tasks"
- **Reality**: Simple tasks (sentiment) can use simpler methods; complex tasks (translation) use LSTM/Transformers

**Mistake 5**: "RNN processes entire sentence at once"
- **Reality**: RNN processes one word at a time, sequentially

**Mistake 6**: "Hidden state is the output"
- **Reality**: Hidden state is internal memory; output comes from output layer applied to hidden state

## 📝 Quick Revision Points

### RNN Architecture

**Basic Structure:**
```
    ┌────────┐
x →│  RNN   │→ h (hidden state)
    │   ⟲   │→ y (output)
    └────────┘
```

**Unfolded:**
```
x₁→[RNN]→h₁→[RNN]→h₂→[RNN]→h₃→[RNN]→h₄
   t=1      t=2      t=3      t=4
```

### Types of RNN

| Type | Input | Output | Example |
|------|-------|--------|---------|
| **One-to-One** | 1 | 1 | Image classification |
| **One-to-Many** | 1 | N | Music generation |
| **Many-to-One** | N | 1 | Sentiment analysis |
| **Many-to-Many** | N | M | Language translation |

### Forward Propagation Equations

**Time Step t:**

$$h_t = \tanh(W \cdot x_t + W_h \cdot h_{t-1} + b)$$

**Output (Many-to-One):**

$$\hat{y} = \sigma(W_y \cdot h_T + b_y)$$

**Output (Many-to-Many):**

$$\hat{y}_t = \sigma(W_y \cdot h_t + b_y)$$

### Weight Matrices

**For Hidden Dimension = 128, Input Dimension = 300:**

| Matrix | Dimension | Purpose |
|--------|-----------|---------|
| **W** | (128 × 300) | Input to hidden |
| **W_h** | (128 × 128) | Hidden to hidden |
| **W_y** | (1 × 128) | Hidden to output (binary) |
| **b** | (128 × 1) | Hidden bias |
| **b_y** | (1 × 1) | Output bias |

### Applications

**One-to-Many:**
- Music generation
- Image captioning
- Text generation

**Many-to-One:**
- Sentiment analysis
- Document classification
- Stock price prediction

**Many-to-Many:**
- Language translation
- Named Entity Recognition (NER)
- Question answering

### Remember

- **RNN = Sequential processing + Memory**
- **Hidden state** = Memory that carries information
- **Shared weights** = Same W across all time steps
- **Forward**: $h_t = f(W \cdot x_t + W_h \cdot h_{t-1} + b)$
- **Vanishing gradient** = Main limitation of RNN
- **LSTM/GRU** = Better alternatives to vanilla RNN
- **Applications** = Text, speech, time series (sequential data)
- **Slower than Transformers** = Cannot parallelize
