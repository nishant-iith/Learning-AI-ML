# Chapter 6: RNN Backpropagation and Introduction to LSTM

## 🎯 Learning Objectives
- Understand backpropagation through time (BPTT) in RNN
- Master weight update formulas using chain rule
- Learn problems with vanilla RNN (vanishing gradient, long-term dependencies)
- Understand why LSTM is needed
- Get introduced to LSTM architecture
- Learn memory cell, forget gate, and input gate concepts

## 📚 Key Concepts

### Backward Propagation in RNN (BPTT)

#### Review: Forward Propagation

**RNN Architecture (Many-to-One):**

```
    ┌────┐      ┌────┐      ┌────┐      ┌────┐
x₁ →│RNN │→ o₁ →│RNN │→ o₂ →│RNN │→ o₃ →│RNN │→ o₄
    └────┘      └────┘      └────┘      └─┬──┘
    t=1         t=2         t=3          t=4
    W           W           W            W
    ↓           ↓           ↓            ↓
               W_h         W_h          W_h
                                         ↓
                                    [Sigmoid/Softmax]
                                         ↓
                                         ŷ
```

**Forward Equations:**

$$o_1 = f(x_1 \cdot W)$$

$$o_2 = f(x_2 \cdot W + o_1 \cdot W_h)$$

$$o_3 = f(x_3 \cdot W + o_2 \cdot W_h)$$

$$o_4 = f(x_4 \cdot W + o_3 \cdot W_h)$$

$$\hat{y} = \sigma(o_4 \cdot W_y)$$

Where:
- $f$ = Activation function (tanh, ReLU)
- $\sigma$ = Sigmoid (binary) or Softmax (multi-class)
- $W$ = Input weight matrix
- $W_h$ = Hidden state weight matrix
- $W_y$ = Output weight matrix

#### Backpropagation Through Time (BPTT)

**Concept**: Gradients flow backward through time steps to update weights

**Flow**:

```
Forward:  x₁ → o₁ → o₂ → o₃ → o₄ → ŷ → Loss
           ↓    ↓    ↓    ↓    ↓    ↓
Backward: ∇x₁ ← ∇o₁ ← ∇o₂ ← ∇o₃ ← ∇o₄ ← ∇ŷ ← ∇L
```

**Key Steps:**

1. **Calculate Loss**
2. **Compute Gradients** (chain rule)
3. **Backpropagate** through time (t=4 → t=3 → t=2 → t=1)
4. **Update Weights**

#### Weight Update Formulas

**General Weight Update Rule:**

$$W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W_{old}}$$

Where:
- $\alpha$ = Learning rate
- $L$ = Loss function
- $\frac{\partial L}{\partial W}$ = Gradient of loss with respect to weight

#### Updating W_h (Hidden State Weights)

**At Time Step t=4:**

**Weight Update:**

$$W_h^{new} = W_h^{old} - \alpha \frac{\partial L}{\partial W_h^{old}}$$

**Gradient Calculation (Chain Rule):**

$$\frac{\partial L}{\partial W_h} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_h}$$

**But $\hat{y}$ depends on $o_4$, and $o_4$ depends on $W_h$:**

$$\frac{\partial L}{\partial W_h} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_4} \cdot \frac{\partial o_4}{\partial W_h}$$

**Detailed Chain:**

Since $o_4 = f(x_4 \cdot W + o_3 \cdot W_h)$:

$$\frac{\partial L}{\partial W_h} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_4} \cdot \frac{\partial o_4}{\partial W_h}$$

#### Updating W (Input Weights)

**At Time Step t=4:**

**Weight Update:**

$$W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W^{old}}$$

**Gradient Calculation:**

Loss depends on $\hat{y}$, which depends on $o_4$, which depends on $W$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_4} \cdot \frac{\partial o_4}{\partial W}$$

**Complete Chain Rule:**

Since $o_4$ depends on $o_3$, $o_3$ depends on $o_2$, $o_2$ depends on $o_1$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_4} \cdot \left( \frac{\partial o_4}{\partial W} + \frac{\partial o_4}{\partial o_3} \cdot \frac{\partial o_3}{\partial W} + \cdots \right)$$

#### Example: Updating W_h at t=3

**Scenario**: Update $W_h$ at time step 3 (between $o_3$ and $o_4$)

**Dependencies:**

```
Loss → ŷ → o₄ → o₃ → W_h (at t=3)
```

**Chain Rule:**

$$\frac{\partial L}{\partial W_h^{(t=3)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_4} \cdot \frac{\partial o_4}{\partial o_3} \cdot \frac{\partial o_3}{\partial W_h}$$

**Breakdown:**

1. $\frac{\partial L}{\partial \hat{y}}$: How loss changes with predicted output
2. $\frac{\partial \hat{y}}{\partial o_4}$: How output changes with hidden state at t=4
3. $\frac{\partial o_4}{\partial o_3}$: How $o_4$ changes with $o_3$
4. $\frac{\partial o_3}{\partial W_h}$: How $o_3$ changes with $W_h$

#### Complete BPTT Algorithm

**Step 1: Forward Pass (Already Done)**

```
x₁ → o₁ → o₂ → o₃ → o₄ → ŷ
```

Calculate all hidden states and output.

**Step 2: Calculate Loss**

$$L = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$$

(For binary classification with Binary Cross-Entropy)

**Step 3: Backward Pass (BPTT)**

```
Time Step 4:
  ∂L/∂W_y, ∂L/∂W_h, ∂L/∂W
↓
Time Step 3:
  ∂L/∂W_h (at t=3), ∂L/∂W (at t=3)
↓
Time Step 2:
  ∂L/∂W_h (at t=2), ∂L/∂W (at t=2)
↓
Time Step 1:
  ∂L/∂W (at t=1)
```

**Step 4: Accumulate Gradients**

Since weights are shared across time steps:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{4} \frac{\partial L}{\partial W^{(t)}}$$

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{4} \frac{\partial L}{\partial W_h^{(t)}}$$

**Step 5: Update Weights**

$$W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W}$$

$$W_h^{new} = W_h^{old} - \alpha \frac{\partial L}{\partial W_h}$$

$$W_y^{new} = W_y^{old} - \alpha \frac{\partial L}{\partial W_y}$$

### Problems with Vanilla RNN

#### 1. Vanishing Gradient Problem

**The Issue:**

During BPTT, gradients are multiplied repeatedly across time steps.

**Mathematical Explanation:**

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial o_T} \cdot \frac{\partial o_T}{\partial o_{T-1}} \cdot \frac{\partial o_{T-1}}{\partial o_{T-2}} \cdots \frac{\partial o_2}{\partial o_1} \cdot \frac{\partial o_1}{\partial W}$$

**Problem with Sigmoid Activation:**

Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$

Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Range of Derivative:**

$$\sigma'(x) \in [0, 0.25]$$

Maximum value = 0.25 (at x = 0)

**Repeated Multiplication:**

```
Time steps = 100

Gradient = 0.25 × 0.25 × 0.25 × ... (100 times)
         = 0.25¹⁰⁰
         ≈ 0 (essentially zero!)
```

**Result:**
- Gradients become exponentially small
- Early time steps get almost zero gradient
- Weights barely update
- Network cannot learn long-term dependencies

**Visual Representation:**

```
Gradient Magnitude Across Time Steps:
t=100  ████████ (1.0)
t=90   ██████   (0.5)
t=80   ████     (0.1)
t=70   ██       (0.01)
t=60   █        (0.001)
t=50            (0.0001)
t=1             (≈ 0)  ← Cannot learn!
```

#### 2. Exploding Gradient Problem

**The Issue:**

Gradients become exponentially large.

**Cause:**

If activation derivatives > 1 and weights large:

```
Gradient = 2 × 2 × 2 × ... (100 times)
         = 2¹⁰⁰
         → ∞ (explodes!)
```

**Result:**
- Weight updates become huge
- Training becomes unstable
- NaN (Not a Number) values appear

**Solution:**
- **Gradient Clipping**: Cap gradients to maximum value
- LSTM architecture

#### 3. Long-Term Dependency Problem

**The Issue:**

RNN cannot capture relationships between words far apart.

**Example 1:**

```
Sentence: "I grew up in France. [100 words...] I speak fluent _____"

Required: "French"

Problem: RNN forgets "France" after 100 words
Result: Cannot predict "French" correctly
```

**Example 2:**

```
Sentence: "Krish likes to eat pizza and he also likes to go to movies"

Words:    [1]   [2]   [3] [4]  [5]   [6]  [7]  [8]   [9]  [10] [11] [12]

Context: "he" (word 7) refers to "Krish" (word 1)
Gap: 6 words
```

**Problem:**
- As gap increases, context is lost
- $o_7$ has forgotten information from $o_1$
- RNN struggles with gaps > 10-20 words

**Why It Happens:**

Each hidden state is:

$$h_t = f(W \cdot x_t + W_h \cdot h_{t-1})$$

Information from $h_1$ must pass through:

$$h_1 \to h_2 \to h_3 \to h_4 \to h_5 \to h_6 \to h_7$$

Each step, information is:
- Transformed by activation function
- Mixed with new input
- Gradually forgotten

**Visual:**

```
Context Retention Over Time:
t=1: ████████████ (100% - "Krish")
t=2: ██████████   (80%)
t=3: ████████     (60%)
t=4: ██████       (40%)
t=5: ████         (20%)
t=6: ██           (10%)
t=7: █            (5%)  ← "he" has forgotten "Krish"!
```

#### 4. Context Information Problem

**Scenario:**

```
Sentence 1: "Krish likes to eat pizza."
Sentence 2: "Yann LeCun likes CNN."
```

**Problem:**

When processing Sentence 2:
- RNN still carries information from Sentence 1
- Context from "Krish" interferes with "Yann LeCun"
- Need to **forget** old context
- Need to **add** new context

**What's Needed:**
- Mechanism to **forget** irrelevant information
- Mechanism to **remember** relevant information
- Mechanism to **add** new context

### Introduction to LSTM

#### Why LSTM?

**LSTM (Long Short-Term Memory)** solves all RNN problems:

✓ **Vanishing Gradient** → Solved with gates and cell state
✓ **Long-Term Dependencies** → Can remember 100+ time steps
✓ **Context Management** → Forgets old, remembers relevant, adds new

#### LSTM Architecture (Basic)

**Comparison:**

**Vanilla RNN Cell:**

```
    ┌────────┐
x →│  tanh  │→ h
    │        │
    └────────┘
```

**LSTM Cell:**

```
         ┌─────────────────────────────┐
    ┌───→│     Memory Cell (C_t)       │───┐
    │    └─────────────────────────────┘   │
    │                 ↓                     │
C_{t-1}         [Forget Gate]               C_t
    │         [Input Gate]                  │
    │         [Output Gate]                 │
    │                 ↓                     │
    └────→  [Operations]  ←─── x_t         │
                     ↓                      ↓
h_{t-1} ────────────────────────────────→ h_t
```

**Key Components:**

1. **Cell State** ($C_t$): Long-term memory
2. **Forget Gate**: Decides what to forget
3. **Input Gate**: Decides what new information to add
4. **Output Gate**: Decides what to output

#### LSTM Components (Detailed)

**Complete LSTM Diagram:**

```
         ┌──────────────── C_{t-1} ────────────────┐
         │                                          │
         ↓                                          ↓
    ┌────────┐        ┌────────┐        ┌────────┐ │
    │Forget  │        │ Input  │        │Output  │ │
    │ Gate   │    ×   │  Gate  │    ×   │ Gate   │ ×
    │(sigmoid│        │(sigmoid│        │(sigmoid│
    └────────┘        └────┬───┘        └────────┘
         ↑                 ↑                 ↑
         │                 │                 │
    [h_{t-1}, x_t]    [h_{t-1}, x_t]   [h_{t-1}, x_t]
                           │
                      ┌────────┐
                      │  tanh  │
                      └────────┘
                           ↓
                    (Candidate values)
                           ↓
                         [+]  ← Add to cell state
                           ↓
                          C_t
                           ↓
                      ┌────────┐
                      │  tanh  │
                      └────────┘
                           ↓
                          h_t
```

#### 1. Cell State (Memory Cell)

**Symbol**: $C_t$

**Purpose**: Long-term memory that runs through entire chain

**Key Features:**
- Information flows through with minimal changes
- Gates can add or remove information
- Solves vanishing gradient problem

**Analogy:**
- Like a conveyor belt carrying information
- Gates decide what to keep, forget, or add

#### 2. Forget Gate

**Symbol**: $f_t$

**Purpose**: Decides what information to **throw away** from cell state

**Equation:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Output**: Value between 0 and 1 for each number in cell state
- **1** = "Keep this completely"
- **0** = "Forget this completely"
- **0.5** = "Keep half of this"

**Example:**

```
Context: "Krish likes pizza."
New Input: "Yann LeCun likes CNN."

Forget Gate: "Forget information about Krish"
→ f_t = 0 (for Krish-related information)
```

**Operation:**

$$C_t = f_t \times C_{t-1}$$

(Pointwise multiplication - × symbol in diagram)

#### 3. Input Gate

**Symbol**: $i_t$

**Purpose**: Decides what **new information** to add to cell state

**Two Parts:**

**Part 1: Input Gate Layer (sigmoid)**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

Decides which values to update (0 = don't update, 1 = update)

**Part 2: Candidate Values (tanh)**

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Creates candidate values to add (-1 to 1)

**Example:**

```
New Input: "Yann LeCun likes CNN"

Input Gate: "Add information about Yann LeCun"
→ i_t = 1 (for new person)
→ C̃_t = [new context vector for "Yann LeCun"]
```

**Update Cell State:**

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

$$C_t = \text{(Forget old)} + \text{(Add new)}$$

#### 4. Output Gate

**Symbol**: $o_t$

**Purpose**: Decides what to **output** based on cell state

**Equation:**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State (Output):**

$$h_t = o_t \times \tanh(C_t)$$

**Explanation:**
- $\tanh(C_t)$: Normalize cell state to [-1, 1]
- $o_t$: Decide which parts to output
- $h_t$: Final hidden state (output of LSTM cell)

#### Complete LSTM Equations

**Summary:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \times \tanh(C_t)$$

#### Operations in LSTM Diagram

**1. Pointwise Multiplication (×):**

```
Symbol: ⊗ or ×

Operation: Element-wise multiplication
Example: [1, 2, 3] × [0.5, 1, 0] = [0.5, 2, 0]
```

**2. Pointwise Addition (+):**

```
Symbol: ⊕ or +

Operation: Element-wise addition
Example: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
```

**3. Concatenation:**

```
Symbol: Two lines joining

Operation: Combine vectors
Example: [h_{t-1}, x_t] = [h₁, h₂, ..., x₁, x₂, ...]
```

#### How LSTM Solves RNN Problems

**1. Vanishing Gradient:**
- Cell state provides direct path for gradient flow
- Additive operation (+) preserves gradients
- No repeated multiplication of small derivatives

**2. Long-Term Dependencies:**
- Cell state acts as "memory highway"
- Information can flow unchanged through many time steps
- Forget/Input gates control what to remember

**3. Context Management:**
- Forget gate: Removes old context
- Input gate: Adds new context
- Output gate: Controls what information to use

### Comparison: RNN vs LSTM

| Aspect | RNN | LSTM |
|--------|-----|------|
| **Memory** | Short-term (10-20 steps) | Long-term (100+ steps) |
| **Vanishing Gradient** | Yes (major problem) | No (solved) |
| **Context Management** | Poor (forgets quickly) | Excellent (gates control) |
| **Parameters** | Fewer | More (4× RNN) |
| **Training Speed** | Faster | Slower |
| **Accuracy** | Lower | Higher |
| **Use Case** | Simple sequences | Complex sequences |

## ❓ Interview Questions & Answers

**Q1: What is Backpropagation Through Time (BPTT)?**

BPTT is the backpropagation algorithm applied to RNN. Process:
1. Unfold RNN across all time steps
2. Calculate loss at output
3. Backpropagate gradients from t=T to t=1
4. Update shared weights using accumulated gradients

**Q2: How do you update weights in RNN using BPTT?**

$$W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W}$$

Gradient calculated using chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \frac{\partial o_T}{\partial W}$$

Since weights are shared across time steps, accumulate gradients:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial W^{(t)}}$$

**Q3: What is the vanishing gradient problem in RNN?**

During BPTT, gradients are multiplied repeatedly:

$$\text{Gradient} = \frac{\partial o_T}{\partial o_{T-1}} \cdot \frac{\partial o_{T-1}}{\partial o_{T-2}} \cdots \frac{\partial o_2}{\partial o_1}$$

With sigmoid activation (derivative ≤ 0.25):

$$0.25^{100} \approx 0$$

Result: Early time steps get zero gradient, cannot learn.

**Q4: What is the exploding gradient problem?**

Opposite of vanishing: gradients become exponentially large.

Cause: Derivative > 1 and large weights:

$$2^{100} \to \infty$$

Result: Unstable training, NaN values.

Solution: Gradient clipping (cap gradients to max value).

**Q5: Why can't RNN handle long-term dependencies?**

Information must pass through many time steps:

$$h_1 \to h_2 \to h_3 \to \cdots \to h_{100}$$

Each step:
- Information transformed by activation
- Mixed with new input
- Gradually forgotten

After 100 steps, $h_1$ information is mostly lost.

**Q6: What is LSTM and why is it better than RNN?**

LSTM (Long Short-Term Memory) has gates and cell state:
- **Forget Gate**: Removes irrelevant information
- **Input Gate**: Adds new information
- **Cell State**: Memory highway (no vanishing gradient)

Solves: Vanishing gradient, long-term dependencies, context management.

**Q7: What is the cell state in LSTM?**

Cell state ($C_t$) is the long-term memory in LSTM:
- Runs through entire chain with minimal changes
- Information flows via additive operations (not multiplicative)
- Gates add or remove information
- Solves vanishing gradient problem

**Q8: What does the forget gate do in LSTM?**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Outputs value between 0-1:
- **1** = Keep this information
- **0** = Forget this information

Applied to cell state: $C_t = f_t \times C_{t-1}$

**Q9: What does the input gate do in LSTM?**

Decides what new information to add:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

Adds new context while forgetting old.

**Q10: Why does LSTM not have vanishing gradient problem?**

Because of cell state with **additive** updates:

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

Gradient flows through addition (+), not multiplication (×):
- Multiplication: Gradients shrink
- Addition: Gradients preserved

Direct path for gradient flow prevents vanishing.

**Q11: How many parameters does LSTM have compared to RNN?**

**RNN**: 2 weight matrices (W, W_h)

**LSTM**: 8 weight matrices:
- Forget gate: $W_f$, $b_f$
- Input gate: $W_i$, $b_i$
- Candidate: $W_C$, $b_C$
- Output gate: $W_o$, $b_o$

LSTM has **4× more parameters** than RNN.

**Q12: When should you use RNN vs LSTM?**

**Use RNN:**
- Short sequences (< 20 time steps)
- Speed critical
- Simple patterns
- Low computational resources

**Use LSTM:**
- Long sequences (100+ time steps)
- Complex dependencies
- Accuracy critical
- Sufficient computational resources

## 💡 Key Takeaways

- **BPTT** = Backpropagation Through Time (backward gradient flow)
- **Weight Update**: $W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W}$
- **Chain Rule**: Used to compute gradients through time steps
- **Vanishing Gradient**: Sigmoid derivative (0-0.25) → repeated multiplication → ≈0
- **Long-Term Dependency**: RNN forgets after 10-20 steps
- **LSTM** = Long Short-Term Memory (solves RNN problems)
- **Cell State** ($C_t$): Long-term memory highway
- **Forget Gate**: Remove irrelevant information (×)
- **Input Gate**: Add new information (+)
- **Output Gate**: Control output
- **LSTM Solves**: Vanishing gradient, long dependencies, context management

## ⚠️ Common Mistakes

**Mistake 1**: "RNN can remember infinite history"
- **Reality**: Vanilla RNN forgets after 10-20 steps due to vanishing gradient

**Mistake 2**: "Backpropagation in RNN is same as feedforward networks"
- **Reality**: BPTT unfolds across time, accumulates gradients from all time steps

**Mistake 3**: "LSTM is always better than RNN"
- **Reality**: LSTM has 4× more parameters, slower training. Use RNN for simple/short sequences

**Mistake 4**: "Forget gate forgets entire cell state"
- **Reality**: Forget gate outputs 0-1 for each dimension, can partially forget

**Mistake 5**: "Cell state and hidden state are the same"
- **Reality**: Cell state ($C_t$) = long-term memory, Hidden state ($h_t$) = output

**Mistake 6**: "Vanishing gradient only happens with sigmoid"
- **Reality**: Also happens with tanh (derivative ≤ 1), but less severe

## 📝 Quick Revision Points

### BPTT Algorithm

**Steps:**
1. Forward pass: Calculate all hidden states
2. Calculate loss
3. Backward pass: Gradient flows t=T → t=1
4. Accumulate gradients across time steps
5. Update weights

### Weight Update

$$W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W}$$

**Chain Rule:**

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \frac{\partial o_T}{\partial W}$$

### RNN Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| **Vanishing Gradient** | Sigmoid derivative (0-0.25) | LSTM gates |
| **Exploding Gradient** | Large weights, derivative > 1 | Gradient clipping, LSTM |
| **Long-Term Dependency** | Information forgotten over time | LSTM cell state |
| **Context Management** | No forget mechanism | LSTM forget/input gates |

### LSTM Gates

**Forget Gate:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate:**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update:**

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

**Output Gate:**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \times \tanh(C_t)$$

### Remember

- **BPTT** = Unfolded RNN + backpropagation
- **Vanishing gradient** = 0.25¹⁰⁰ ≈ 0
- **LSTM cell state** = Memory highway
- **Forget gate** = Remove old context
- **Input gate** = Add new context
- **Output gate** = Control output
- **LSTM parameters** = 4× RNN parameters
- **Use LSTM** = Long sequences, complex dependencies
