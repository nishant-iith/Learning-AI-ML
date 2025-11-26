# Chapter 11: Transformers - Attention Is All You Need

## 🎯 Learning Objectives
- Understand Transformer architecture and motivation
- Master self-attention mechanism
- Learn multi-head attention concept
- Understand positional encoding
- Know encoder-decoder structure in Transformers
- Compare Transformers with RNN/LSTM
- Understand why Transformers revolutionized NLP

## 📚 Key Concepts

### Introduction to Transformers

#### The Revolution: "Attention Is All You Need"

**Research Paper**: "Attention Is All You Need" (2017)
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **Institution**: Google Brain + University of Toronto
- **Impact**: State-of-the-art in NLP, foundation for BERT, GPT, T5

**Key Innovation**: Replaced RNN/LSTM with attention-only architecture

#### Why Transformers?

**Problems with RNN/LSTM:**

1. **Sequential Processing**:
   - Cannot parallelize (must process word-by-word)
   - Slow training
   - Inefficient use of GPUs

2. **Long-Range Dependencies**:
   - Vanishing gradient problem
   - Forgets early information in long sequences

3. **Fixed Context**:
   - Information bottleneck in encoder-decoder

**Transformer Solutions:**

1. **Parallel Processing**:
   - Processes all words simultaneously
   - Fast training
   - GPU-efficient

2. **Direct Connections**:
   - Attention mechanism connects any two positions
   - No vanishing gradient
   - Perfect memory of all positions

3. **Flexible Context**:
   - Attention weights determine relevance
   - Dynamic context based on input

### High-Level Architecture

#### Transformer as Black Box

**Input**: Sentence in language A (e.g., French)
**Output**: Sentence in language B (e.g., English)

```
Input: "Je suis étudiant"
         ↓
    [Transformer]
         ↓
Output: "I am a student"
```

#### Inside the Black Box: Encoder-Decoder

```
Input → [Encoders] → [Decoders] → Output
```

**Encoder**: Processes input sequence
**Decoder**: Generates output sequence

#### Stack of Encoders and Decoders

**6 Encoder Layers** (stacked):

```
Input
  ↓
[Encoder 1]
  ↓
[Encoder 2]
  ↓
[Encoder 3]
  ↓
[Encoder 4]
  ↓
[Encoder 5]
  ↓
[Encoder 6]
  ↓
Context Vector
```

**6 Decoder Layers** (stacked):

```
Context Vector + Previous Output
  ↓
[Decoder 1]
  ↓
[Decoder 2]
  ↓
[Decoder 3]
  ↓
[Decoder 4]
  ↓
[Decoder 5]
  ↓
[Decoder 6]
  ↓
Output Word
```

**Why 6 layers?**
- Hyperparameter (can be tuned)
- Research paper found 6 optimal for many tasks
- More layers = more capacity, slower training

### Encoder Architecture

#### Single Encoder Block

```
Input
  ↓
[Self-Attention Layer]
  ↓
[Add & Normalize]
  ↓
[Feed-Forward Neural Network]
  ↓
[Add & Normalize]
  ↓
Output to Next Encoder
```

**Two Main Components:**
1. **Self-Attention Layer**: Captures relationships between words
2. **Feed-Forward Neural Network**: Processes each position independently

**Additional Components:**
- Residual connections (Add)
- Layer normalization (Normalize)

#### Input Embeddings

**Step 1: Convert words to vectors**

```
Word: "thinking"
  ↓
Word2Vec/Embedding
  ↓
Vector: [512 dimensions]
```

**Embedding Dimension**: 512 (hyperparameter)
- Each word → 512-dimensional vector
- Learnable during training

#### Positional Encoding

**Problem**: Transformer processes all words in parallel
- Cannot distinguish position (word order)
- "dog bites man" vs "man bites dog" would look same!

**Solution**: Add position information to embeddings

**Formula** (sinusoidal):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in sequence
- $i$ = dimension index
- $d_{model}$ = 512 (embedding dimension)

**Visual:**

```
Word Embedding:      [0.5, 0.3, -0.2, 0.7, ...]  (512 dims)
Positional Encoding: [0.1, 0.9,  0.5, -0.3, ...]  (512 dims)
                    ⊕  (element-wise addition)
Final Input:         [0.6, 1.2,  0.3, 0.4, ...]  (512 dims)
```

**Why sinusoidal?**
- Unique pattern for each position
- Model can learn relative positions
- Can generalize to longer sequences than seen in training

### Self-Attention Mechanism

#### Core Idea

**Question**: How much should each word "attend" to every other word?

**Example**:

```
Sentence: "The animal didn't cross the street because it was too tired"

At word "it":
- Should attend to "animal" (high attention)
- Should attend to "tired" (medium attention)
- Should attend to "the" (low attention)
```

**Goal**: Compute attention weights for all word pairs

#### Self-Attention Step-by-Step

**Input**: Sentence with words $x_1, x_2, ..., x_n$

Each word embedded as vector (512 dims)

**Step 1: Create Q, K, V Matrices**

**Three weight matrices** (learned during training):
- $W^Q$ (Query weights): 512 × 64
- $W^K$ (Key weights): 512 × 64
- $W^V$ (Value weights): 512 × 64

**Multiply each word embedding**:

$$Q = X \cdot W^Q \quad \text{(Queries)}$$

$$K = X \cdot W^K \quad \text{(Keys)}$$

$$V = X \cdot W^V \quad \text{(Values)}$$

**Dimensions**:
- Input $X$: (sequence_length, 512)
- $Q$, $K$, $V$: (sequence_length, 64)

**Example with 2 words**:

```
Word 1: "thinking" → x₁ (512 dims)
Word 2: "machines" → x₂ (512 dims)

Q₁ = x₁ · W^Q  →  q₁ (64 dims)
K₁ = x₁ · W^K  →  k₁ (64 dims)
V₁ = x₁ · W^V  →  v₁ (64 dims)

Q₂ = x₂ · W^Q  →  q₂ (64 dims)
K₂ = x₂ · W^K  →  k₂ (64 dims)
V₂ = x₂ · W^V  →  v₂ (64 dims)
```

**Intuition**:
- **Query**: What am I looking for?
- **Key**: What do I offer?
- **Value**: What information do I contain?

**Step 2: Compute Attention Scores**

**Score** = Similarity between queries and keys

$$\text{Score}(q_i, k_j) = q_i \cdot k_j^T$$

**Example**:

```
Score for "thinking" attending to itself:
score₁₁ = q₁ · k₁^T = 112

Score for "thinking" attending to "machines":
score₁₂ = q₁ · k₂^T = 96
```

**Step 3: Scale Scores**

$$\text{Scaled Score} = \frac{\text{Score}}{\sqrt{d_k}}$$

Where $d_k = 64$ (dimension of keys)

$$\sqrt{d_k} = \sqrt{64} = 8$$

**Example**:

```
Scaled score₁₁ = 112 / 8 = 14
Scaled score₁₂ = 96 / 8 = 12
```

**Why scale?**
- Prevents softmax saturation for large $d_k$
- Stabilizes gradients
- Hyperparameter (found empirically)

**Step 4: Apply Softmax**

$$\text{Attention Weights} = \text{softmax}(\text{Scaled Scores})$$

**Example**:

```
Softmax([14, 12]) = [0.88, 0.12]
```

**Interpretation**:
- "thinking" attends 88% to itself
- "thinking" attends 12% to "machines"

**Step 5: Weighted Sum of Values**

$$\text{Output} = \sum_j \text{Attention Weight}_j \times v_j$$

**Example**:

```
z₁ = 0.88 × v₁ + 0.12 × v₂
```

**Complete Formula**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Matrix Form (All Words at Once)

**Input**:
- $Q$: (seq_len, 64)
- $K$: (seq_len, 64)
- $V$: (seq_len, 64)

**Computation**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Output**: (seq_len, 64)

**Parallelization**: All words processed simultaneously!

### Multi-Head Attention

#### Limitation of Single Attention Head

**Problem**: Single attention head focuses on one aspect

**Example**: "it" in sentence
- Single head: Focuses mainly on "animal"
- Misses: "tired" (also relevant)

**Solution**: Multiple attention heads

#### Multi-Head Attention Concept

**Use 8 different attention heads** (hyperparameter)

Each head:
- Has own $W^Q$, $W^K$, $W^V$ weights
- Learns different relationships
- Focuses on different aspects

**Head 0**: Subject-verb relationships
**Head 1**: Object relationships
**Head 2**: Adjective-noun relationships
...
**Head 7**: Other patterns

#### Multi-Head Attention Formula

**For each head $h$**:

$$\text{head}_h = \text{Attention}(QW_h^Q, KW_h^K, VW_h^V)$$

**Concatenate all heads**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, ..., \text{head}_7)W^O$$

Where:
- 8 heads (hyperparameter)
- Each head: 64 dimensions
- Concatenated: 8 × 64 = 512 dimensions
- $W^O$: Output weight matrix (512 × 512)

#### Visualization

**Example**: "it" attending to other words

```
Head 0:  [animal: ■■■■■■■■ (88%), tired: ■ (12%)]
Head 1:  [animal: ■■■ (40%), tired: ■■■■■■ (60%)]
Head 2:  [animal: ■■■■■ (55%), street: ■■■ (45%)]
...
Head 7:  [animal: ■■■■ (50%), cross: ■■■■ (50%)]
```

**Combined**: Captures multiple relationships!

### Feed-Forward Neural Network

**After self-attention**, each position passes through FFN

**Architecture**:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**Two layers**:
1. **Layer 1**: 512 → 2048 (expansion) with ReLU
2. **Layer 2**: 2048 → 512 (compression)

**Applied independently** to each position

**Why FFN?**
- Adds non-linearity
- Processes information after attention
- Increases model capacity

### Residual Connections and Layer Normalization

#### Residual Connection (Skip Connection)

**Concept**: Add input to output

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Visual**:

```
Input x ──────────────┐
     ↓                │
[Self-Attention]      │
     ↓                │
   output          (skip)
     ↓                │
   [+ ]←──────────────┘
     ↓
[LayerNorm]
     ↓
   Final Output
```

**Benefits**:
- Easier gradient flow
- Prevents vanishing gradients
- Allows very deep networks

#### Layer Normalization

**Normalize across features** (not batch)

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

**Benefits**:
- Stabilizes training
- Reduces internal covariate shift
- Improves convergence

### Decoder Architecture

#### Single Decoder Block

```
Previous Output
  ↓
[Masked Self-Attention]
  ↓
[Add & Normalize]
  ↓
[Encoder-Decoder Attention]  ← Encoder Output
  ↓
[Add & Normalize]
  ↓
[Feed-Forward Network]
  ↓
[Add & Normalize]
  ↓
Output to Next Decoder
```

**Three Main Components**:
1. **Masked Self-Attention**: Prevents attending to future positions
2. **Encoder-Decoder Attention**: Attends to encoder output
3. **Feed-Forward Network**: Same as encoder

#### Masked Self-Attention

**Problem**: During training, decoder sees full target sequence
- Would "cheat" by looking at future words

**Solution**: Mask future positions

**Masking**:

```
Generating: "I am a student"

At position 2 ("am"):
- Can attend to: "I", "am"
- Cannot attend to: "a", "student"

Attention Matrix:
     I    am    a   student
I   [0.5  0     0     0   ]
am  [0.3  0.7   0     0   ]  ← Masked (future = 0)
a   [0.1  0.2  0.7    0   ]
stu [0.1  0.1  0.2   0.6  ]
```

**Implementation**: Set future positions to $-\infty$ before softmax

#### Encoder-Decoder Attention

**Different from self-attention**:
- **Queries (Q)**: From decoder
- **Keys (K)**: From encoder output
- **Values (V)**: From encoder output

**Purpose**: Decoder attends to relevant parts of input sequence

**Example**:

```
Input (French): "Je suis étudiant"
Decoder generating: "I am a ___"

At "am":
- Attends strongly to "suis" (French for "am")
- Attends weakly to "Je", "étudiant"
```

#### Sequential Generation

**Unlike encoder**, decoder generates **one word at a time**

**Process**:

```
Time Step 1:
  Input: <START>
  Output: "I"

Time Step 2:
  Input: "I"
  Output: "am"

Time Step 3:
  Input: "I am"
  Output: "a"

Time Step 4:
  Input: "I am a"
  Output: "student"

Time Step 5:
  Input: "I am a student"
  Output: <END>
```

**Why sequential?**
- Cannot know future output
- Each word depends on previous words
- Generation is inherently sequential

### Output Layer

#### Linear + Softmax

**After final decoder**:

```
Decoder Output (512 dims)
        ↓
[Linear Layer]  (512 → vocab_size)
        ↓
[Softmax]
        ↓
Probability Distribution over Vocabulary
        ↓
Select Word with Highest Probability
```

**Example**:

```
Vocabulary: ["I", "am", "a", "student", "the", "is", ...]  (30,000 words)

Linear output: [2.3, 1.7, 0.5, 3.1, 0.2, ...]
              ↓
Softmax: [0.05, 0.03, 0.01, 0.12, 0.01, ...]
              ↓
Argmax: "student" (index 3, probability 0.12)
```

### Complete Transformer Architecture

```
Input Sequence (French)
       ↓
[Input Embedding + Positional Encoding]
       ↓
    ┌─────────────┐
    │  Encoder 1  │ ← [Self-Attention + FFN]
    ├─────────────┤
    │  Encoder 2  │
    ├─────────────┤
    │     ...     │
    ├─────────────┤
    │  Encoder 6  │
    └─────┬───────┘
          │ Encoder Output
          │
          ├─────────────────────────┐
          ↓                         ↓
Previous Output (English)    Encoder Output
       ↓                          │
[Output Embedding + Positional Encoding]
       ↓                          │
    ┌─────────────────────────────┴─┐
    │  Decoder 1  │ ← [Masked Self-Attn + Enc-Dec Attn + FFN]
    ├─────────────┤
    │  Decoder 2  │
    ├─────────────┤
    │     ...     │
    ├─────────────┤
    │  Decoder 6  │
    └─────┬───────┘
          ↓
   [Linear + Softmax]
          ↓
    Output Word
```

## Comparison: Transformer vs RNN/LSTM

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **Processing** | Sequential (one word at a time) | Parallel (all words at once) |
| **Training Speed** | Slow (sequential) | Fast (parallelizable) |
| **Long Dependencies** | Struggles (vanishing gradient) | Excellent (direct attention) |
| **GPU Utilization** | Poor (sequential) | Excellent (parallel) |
| **Memory** | Hidden state bottleneck | Full attention matrix |
| **Position Awareness** | Implicit (through sequence) | Explicit (positional encoding) |
| **Complexity** | O(n) per layer | O(n²) (attention matrix) |
| **Best For** | Small sequences, streaming | Long sequences, batch processing |

## Hyperparameters in Transformer

| Hyperparameter | Value (Base Model) | Purpose |
|----------------|-------------------|---------|
| **d_model** | 512 | Embedding dimension |
| **Number of Layers** | 6 | Encoder/decoder depth |
| **Number of Heads** | 8 | Multi-head attention |
| **d_k** (Key dim) | 64 | Query/key dimension |
| **d_v** (Value dim) | 64 | Value dimension |
| **d_ff** | 2048 | FFN hidden layer size |
| **Dropout** | 0.1 | Regularization |
| **Vocabulary Size** | 30,000-50,000 | Number of words |

## ❓ Interview Questions & Answers

**Q1: What is the main innovation of Transformers?**

Replaced sequential processing (RNN/LSTM) with **attention-only** architecture:
- Processes all words in parallel
- Uses self-attention to capture relationships
- No recurrence or convolution

**Q2: What is self-attention?**

Mechanism that computes attention weights between all pairs of words in a sequence:
- Each word attends to every other word
- Determines which words are most relevant
- Formula: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$

**Q3: What are Q, K, V in self-attention?**

**Query (Q)**: "What am I looking for?"
**Key (K)**: "What do I offer?"
**Value (V)**: "What information do I contain?"

Created by multiplying input with weight matrices $W^Q$, $W^K$, $W^V$

**Q4: Why divide by √d_k in attention formula?**

Prevents softmax saturation for large dimensions:
- Without scaling: Large dot products → extreme softmax values
- With scaling: Stable gradients, better training
- Empirically found to work well

**Q5: What is multi-head attention?**

Uses multiple attention heads (typically 8):
- Each head learns different relationships
- Heads have separate $W^Q$, $W^K$, $W^V$ weights
- Outputs concatenated and projected
- Captures diverse patterns

**Q6: Why do we need positional encoding?**

Transformers process all words in parallel:
- No inherent position information
- "dog bites man" ≠ "man bites dog"
- Positional encoding adds position information
- Sine/cosine functions create unique position patterns

**Q7: What is the difference between encoder and decoder?**

**Encoder**:
- Bidirectional (sees full input sequence)
- Self-attention not masked
- Processes input sequence

**Decoder**:
- Unidirectional (cannot see future)
- Masked self-attention
- Generates output sequentially
- Has encoder-decoder attention

**Q8: Why is decoder self-attention masked?**

Prevents "cheating" during training:
- Without masking: Sees future words
- Would learn to copy future words
- Masked: Only sees previous words
- Simulates real generation scenario

**Q9: What is encoder-decoder attention?**

Attention in decoder where:
- **Queries**: From decoder
- **Keys & Values**: From encoder output
- Decoder attends to relevant input parts
- Enables translation/summarization

**Q10: How does Transformer handle variable-length sequences?**

**Padding** + **Masking**:
- Pad shorter sequences to max length
- Mask padded positions in attention
- Prevents attending to padding
- Handles any sequence length

**Q11: What are residual connections in Transformers?**

Skip connections that add input to output:
- $\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$
- Easier gradient flow
- Enables very deep networks (many layers)

**Q12: Why are Transformers better than RNNs for long sequences?**

1. **No vanishing gradient**: Direct attention connections
2. **Parallel processing**: All positions at once
3. **Perfect memory**: Attention to any position
4. **Faster training**: GPU-efficient parallelization

## 💡 Key Takeaways

- **Transformers** = Attention-only architecture (no RNN/CNN)
- **Parallel Processing** = All words processed simultaneously
- **Self-Attention** = Words attend to all other words
- **Multi-Head Attention** = 8 heads capture different relationships
- **Positional Encoding** = Sine/cosine functions add position info
- **Encoder** = 6 layers of self-attention + FFN
- **Decoder** = 6 layers of masked self-attention + encoder-decoder attention + FFN
- **Q, K, V** = Query, Key, Value matrices from input embeddings
- **Attention Formula**: $\text{softmax}(QK^T/\sqrt{d_k})V$
- **Applications** = Translation, summarization, BERT, GPT, T5

## ⚠️ Common Mistakes

**Mistake 1**: "Transformers are RNNs with attention"
- **Reality**: No recurrence at all, pure attention-based

**Mistake 2**: "Self-attention and multi-head attention are same"
- **Reality**: Multi-head uses multiple self-attention heads

**Mistake 3**: "Encoder and decoder process sequentially"
- **Reality**: Encoder parallel, decoder sequential (during generation)

**Mistake 4**: "Q, K, V are different inputs"
- **Reality**: All derived from same input with different weight matrices

**Mistake 5**: "Positional encoding is learned"
- **Reality**: Fixed sinusoidal functions (not learned in original paper)

**Mistake 6**: "Transformers always better than RNNs"
- **Reality**: RNNs better for streaming, small sequences, less memory

## 📝 Quick Revision Points

### Architecture Summary

**Encoder (×6)**:
1. Input Embedding + Positional Encoding
2. Multi-Head Self-Attention
3. Add & LayerNorm
4. Feed-Forward Network
5. Add & LayerNorm

**Decoder (×6)**:
1. Output Embedding + Positional Encoding
2. Masked Multi-Head Self-Attention
3. Add & LayerNorm
4. Multi-Head Encoder-Decoder Attention
5. Add & LayerNorm
6. Feed-Forward Network
7. Add & LayerNorm

**Output**:
- Linear (512 → vocab_size)
- Softmax

### Self-Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Key Dimensions

- **Embedding** ($d_{model}$): 512
- **Queries/Keys** ($d_k$): 64
- **Values** ($d_v$): 64
- **FFN Hidden**: 2048
- **Attention Heads**: 8

### Advantages

✓ Parallel processing
✓ No vanishing gradient
✓ Long-range dependencies
✓ GPU-efficient
✓ State-of-the-art results

### Disadvantages

✗ O(n²) memory (attention matrix)
✗ Cannot process streaming data
✗ More complex than RNNs
✗ Requires more data

### Remember

- **"Attention Is All You Need"** = No RNN/CNN, only attention
- **Encoder** = Processes input in parallel
- **Decoder** = Generates output sequentially
- **Q, K, V** = From same input, different weights
- **Multi-head** = 8 different attention heads
- **Positional encoding** = Sine/cosine position signals
- **Masking** = Prevents seeing future in decoder
- **Residual** = Skip connections for deep networks
