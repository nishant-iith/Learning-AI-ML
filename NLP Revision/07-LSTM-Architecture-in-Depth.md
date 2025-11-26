# Chapter 7: LSTM Architecture in Depth

## 🎯 Learning Objectives
- Understand detailed LSTM architecture layer by layer
- Master memory cell concept (conveyor belt analogy)
- Learn forget gate, input gate, and output gate mechanics
- Understand context switching in sentences
- Know visual notation for LSTM diagrams
- Master complete LSTM flow with examples

## 📚 Key Concepts

### Visual Notation for LSTM Diagrams

**Understanding LSTM diagrams requires knowing the symbols:**

#### 1. Activation Function Boxes

**Sigmoid Box (σ):**

```
┌────────┐
│   σ    │  → Neural network with sigmoid activation
└────────┘
```

Output range: [0, 1]
- **0** = Complete rejection/forgetting
- **1** = Complete acceptance/keeping
- **0.5** = Partial acceptance

**Tanh Box:**

```
┌────────┐
│  tanh  │  → Neural network with tanh activation
└────────┘
```

Output range: [-1, 1]
- Creates candidate values
- Normalizes information

#### 2. Line Operations

**Concatenation (Two lines joining):**

```
Line 1 ─┐
        ├─→ Combined vector
Line 2 ─┘
```

Operation: Combine two vectors into one
Example: [h_{t-1}, x_t] = Concatenate hidden state and input

**Copy (Lines splitting):**

```
     ┌─→ Copy 1
Line─┤
     └─→ Copy 2
```

Operation: Duplicate information to multiple paths

**Vector Transform (Arrow):**

```
───→
```

Operation: Information flows from one component to another

#### 3. Pointwise Operations (Circle)

**Pointwise Multiplication (×):**

```
 ⊗  or  ×
```

Operation: Element-wise multiplication
Example: [1, 2, 3] × [0.5, 1, 0] = [0.5, 2, 0]

**Pointwise Addition (+):**

```
 ⊕  or  +
```

Operation: Element-wise addition
Example: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]

### RNN vs LSTM: Visual Comparison

#### Traditional RNN Cell

```
         ┌──────────┐
h_{t-1} →│          │
         │   tanh   │→ h_t
x_t ────→│          │
         └──────────┘

Single tanh layer
Simple structure
```

**Issues:**
- Short-term memory only
- Vanishing gradient
- No context control

#### LSTM Cell

```
                C_{t-1}
                  ↓
         ┌────────────────────┐
         │                    │
         │    Memory Cell     │
         │                    │
         └────────────────────┘
         ↓     ↓      ↓      ↓
      [Forget][Input][Output]
         ↓     ↓      ↓      ↓
h_{t-1}, x_t combined with all gates
         ↓
        h_t, C_t

Multiple layers with gates
Complex structure
```

**Advantages:**
- Long-term memory
- No vanishing gradient
- Context control via gates

### Memory Cell (C_t)

#### The Conveyor Belt Analogy

**Airport Luggage Conveyor Belt:**

```
Luggage →  [═══════════════════]  → Luggage out
           ↑                   ↑
         Add luggage     Remove luggage
```

**Memory Cell:**

```
Info →  [C_{t-1} ════════ C_t]  → Info out
        ↑                    ↑
    Add new info        Forget old info
```

**Key Properties:**

1. **Add Information**: New context can be added
2. **Remove Information**: Old context can be removed
3. **Continuous Flow**: Information flows through time
4. **Minimal Changes**: Only gates modify the information

#### Memory Cell Operations

**Cell State Update:**

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

**Breakdown:**
- $C_{t-1}$: Previous cell state (old memory)
- $f_t$: Forget gate output (what to forget)
- $f_t \times C_{t-1}$: Remove old information
- $i_t \times \tilde{C}_t$: Add new information
- $C_t$: Updated cell state (new memory)

**Visual Flow:**

```
Old Memory (C_{t-1})
       ↓
    [× f_t]  ← Forget gate (remove info)
       ↓
[+ i_t × C̃_t]  ← Input gate (add info)
       ↓
New Memory (C_t)
```

### Forget Gate Layer

#### Purpose

**Decides what information to throw away from cell state**

**Key Question**: "Should I forget this old context?"

#### Forget Gate Equation

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Where:
- $f_t$ = Forget gate activation (0 to 1)
- $\sigma$ = Sigmoid function
- $W_f$ = Weight matrix for forget gate
- $[h_{t-1}, x_t]$ = Concatenation of previous hidden state and current input
- $b_f$ = Bias term

**Output Interpretation:**
- $f_t = 0$ → Forget completely
- $f_t = 1$ → Remember completely
- $f_t = 0.5$ → Remember half

#### Example 1: No Context Switch

**Sentence**: "Krish likes pizza but he does not like burger"

**Analysis:**

```
Words:    Krish  likes  pizza  but  he  does  not  like  burger
Context:  [─────────────── Same person (Krish) ───────────────]
```

**At word "he":**
- Previous context: "Krish likes pizza"
- Current word: "he" (refers to Krish)
- **Context switch?** NO

**Forget Gate Behavior:**

$$h_{t-1} = \text{[info about "Krish likes pizza"]}$$

$$x_t = \text{[vector for "he"]}$$

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \approx [0.9, 0.95, 0.88, ...]$$

**Result**: Values close to **1** → Keep previous context

**Memory Cell Update:**

$$C_t = [0.9, 0.95, 0.88, ...] \times C_{t-1}$$

Almost all information from $C_{t-1}$ is retained!

#### Example 2: Context Switch

**Sentence**: "Krish likes pizza but his friend likes burger"

**Analysis:**

```
Words:    Krish  likes  pizza  but  his  friend  likes  burger
Context:  [──── Krish ────]   [──── friend ────]
                            ↑
                    Context switch!
```

**At word "friend":**
- Previous context: "Krish likes pizza"
- Current word: "friend" (new person!)
- **Context switch?** YES

**Forget Gate Behavior:**

$$h_{t-1} = \text{[info about "Krish likes pizza"]}$$

$$x_t = \text{[vector for "friend"]}$$

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \approx [0.1, 0.05, 0.15, ...]$$

**Result**: Values close to **0** → Forget previous context

**Memory Cell Update:**

$$C_t = [0.1, 0.05, 0.15, ...] \times C_{t-1} \approx [0, 0, 0, ...]$$

Most information from $C_{t-1}$ is **forgotten**!

#### Forget Gate Visualization

```
No Context Switch:
C_{t-1} = [5, 7, 9, 3]
f_t     = [0.9, 0.95, 0.88, 0.92]
         ×  (pointwise)
Result  = [4.5, 6.65, 7.92, 2.76]  ← Most info retained

Context Switch:
C_{t-1} = [5, 7, 9, 3]
f_t     = [0.1, 0.05, 0.15, 0.08]
         ×  (pointwise)
Result  = [0.5, 0.35, 1.35, 0.24]  ← Most info forgotten
```

### Input Gate Layer

#### Purpose

**Decides what new information to add to cell state**

**Key Question**: "What new context should I remember?"

#### Input Gate Has Two Parts

**Part 1: Input Gate (Sigmoid)**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

Decides **which values** to update (0 = don't update, 1 = update)

**Part 2: Candidate Values (Tanh)**

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Creates **new candidate values** to add (range: -1 to 1)

**Combined Operation:**

$$\text{New info to add} = i_t \times \tilde{C}_t$$

#### Example: Adding New Context

**Sentence**: "Krish likes pizza but his friend likes burger"

**At word "friend":**

**Step 1: Input Gate (What to update)**

$$i_t = \sigma(W_i \cdot [h_{t-1}, \text{friend}] + b_i) \approx [0.95, 0.88, 0.92, ...]$$

**Interpretation**: Update most dimensions (new person = new context)

**Step 2: Candidate Values (What values)**

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, \text{friend}] + b_C) \approx [0.7, -0.3, 0.5, ...]$$

**Interpretation**: New context representation for "friend"

**Step 3: Combine**

$$i_t \times \tilde{C}_t = [0.95, 0.88, 0.92, ...] \times [0.7, -0.3, 0.5, ...]$$

$$= [0.665, -0.264, 0.46, ...]$$

**Result**: New information ready to add to memory cell

#### Input Gate Visualization

```
Example: Context switch to "friend"

Input Gate (i_t):
  [0.95, 0.88, 0.92, 0.91]  ← Which dimensions to update

Candidate Values (C̃_t):
  [0.7, -0.3, 0.5, 0.8]     ← New context values

Pointwise multiply:
  [0.665, -0.264, 0.46, 0.728]  ← New info to add
```

#### Why Two Parts?

**Input gate (sigmoid)**: Acts as a **selector**
- Controls which dimensions to update
- Similar to forget gate behavior

**Candidate values (tanh)**: Provides **new information**
- Creates actual new context representation
- Bounded values prevent explosion

**Together**: Selective addition of new information

### Complete Cell State Update

#### Combining Forget and Input

**Full Equation:**

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

$$C_t = \text{(Forget old)} + \text{(Add new)}$$

#### Example: Full Update

**Scenario**: "Krish likes pizza but his friend likes burger"

**At word "friend" (context switch):**

**Step 1: Forget Gate (Remove old)**

$$C_{t-1} = [5, 7, 9, 3]$$

$$f_t = [0.1, 0.05, 0.15, 0.08]$$

$$f_t \times C_{t-1} = [0.5, 0.35, 1.35, 0.24]$$

**Step 2: Input Gate (Add new)**

$$i_t = [0.95, 0.88, 0.92, 0.91]$$

$$\tilde{C}_t = [0.7, -0.3, 0.5, 0.8]$$

$$i_t \times \tilde{C}_t = [0.665, -0.264, 0.46, 0.728]$$

**Step 3: Combine**

$$C_t = [0.5, 0.35, 1.35, 0.24] + [0.665, -0.264, 0.46, 0.728]$$

$$C_t = [1.165, 0.086, 1.81, 0.968]$$

**Result**: Old context (Krish) mostly removed, new context (friend) added!

#### Visualization

```
Before (C_{t-1}):
  [5.0, 7.0, 9.0, 3.0]  ← Krish context
       ↓
  [× forget gate]
       ↓
  [0.5, 0.35, 1.35, 0.24]  ← Most removed
       ↓
  [+ new info]
       ↓
After (C_t):
  [1.165, 0.086, 1.81, 0.968]  ← Friend context
```

### Output Gate Layer

#### Purpose

**Decides what to output based on cell state**

**Key Question**: "What information should I send to next cell?"

#### Output Gate Equations

**Part 1: Output Gate (Sigmoid)**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

Decides which parts of cell state to output

**Part 2: Hidden State (Output)**

$$h_t = o_t \times \tanh(C_t)$$

Combines output gate with normalized cell state

#### Why Tanh on Cell State?

$$\tanh(C_t)$$

**Purpose**: Normalize cell state values to [-1, 1]

**Reason**: Cell state can grow large over time
- Prevents exploding values
- Ensures bounded output
- Makes output gate effective

#### Example: Output Computation

**At word "friend":**

**Cell state after update:**

$$C_t = [1.165, 0.086, 1.81, 0.968]$$

**Step 1: Normalize cell state**

$$\tanh(C_t) = [\tanh(1.165), \tanh(0.086), \tanh(1.81), \tanh(0.968)]$$

$$\approx [0.82, 0.085, 0.95, 0.75]$$

**Step 2: Output gate**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \approx [0.9, 0.85, 0.95, 0.88]$$

**Step 3: Final hidden state**

$$h_t = o_t \times \tanh(C_t)$$

$$h_t = [0.9, 0.85, 0.95, 0.88] \times [0.82, 0.085, 0.95, 0.75]$$

$$h_t = [0.738, 0.072, 0.903, 0.66]$$

**Result**: This $h_t$ is sent to next LSTM cell and used for prediction

#### Output Gate Visualization

```
Cell State (C_t):
  [1.165, 0.086, 1.81, 0.968]
       ↓
  [tanh normalization]
       ↓
  [0.82, 0.085, 0.95, 0.75]
       ↓
  [× output gate]
       ↓
Hidden State (h_t):
  [0.738, 0.072, 0.903, 0.66]
       ↓
  [To next cell & prediction]
```

### Complete LSTM Flow: Step-by-Step

#### Sentence: "Krish likes pizza but his friend likes burger"

**Predicting**: Next word after "burger"

#### Time Step 1: "Krish"

**Inputs:**
- $h_0 = [0, 0, 0, 0]$ (initial)
- $x_1 = \text{Word2Vec("Krish")}$
- $C_0 = [0, 0, 0, 0]$ (initial)

**Forget Gate:**

$$f_1 = \sigma(W_f \cdot [h_0, x_1] + b_f) \approx [0.5, 0.5, 0.5, 0.5]$$

(Random initialization, no previous context)

**Input Gate:**

$$i_1 = \sigma(W_i \cdot [h_0, x_1] + b_i) \approx [0.9, 0.85, 0.88, 0.92]$$

$$\tilde{C}_1 = \tanh(W_C \cdot [h_0, x_1] + b_C) \approx [0.6, 0.7, 0.5, 0.8]$$

**Cell State Update:**

$$C_1 = [0.5, 0.5, 0.5, 0.5] \times [0, 0, 0, 0] + [0.9, 0.85, 0.88, 0.92] \times [0.6, 0.7, 0.5, 0.8]$$

$$C_1 = [0, 0, 0, 0] + [0.54, 0.595, 0.44, 0.736]$$

$$C_1 = [0.54, 0.595, 0.44, 0.736]$$

**Output Gate:**

$$o_1 = \sigma(W_o \cdot [h_0, x_1] + b_o) \approx [0.85, 0.9, 0.88, 0.87]$$

$$h_1 = [0.85, 0.9, 0.88, 0.87] \times \tanh([0.54, 0.595, 0.44, 0.736])$$

$$h_1 \approx [0.39, 0.48, 0.34, 0.54]$$

#### Time Step 5: "friend" (Context Switch!)

**Inputs:**
- $h_4 = [0.7, 0.8, 0.6, 0.75]$ (from "his")
- $x_5 = \text{Word2Vec("friend")}$
- $C_4 = [5, 7, 9, 3]$ (contains Krish context)

**Forget Gate:**

$$f_5 = \sigma(W_f \cdot [h_4, x_5] + b_f) \approx [0.1, 0.05, 0.15, 0.08]$$

(Low values → Forget Krish context!)

**Input Gate:**

$$i_5 = \sigma(W_i \cdot [h_4, x_5] + b_i) \approx [0.95, 0.88, 0.92, 0.91]$$

$$\tilde{C}_5 = \tanh(W_C \cdot [h_4, x_5] + b_C) \approx [0.7, -0.3, 0.5, 0.8]$$

(New friend context!)

**Cell State Update:**

$$C_5 = [0.1, 0.05, 0.15, 0.08] \times [5, 7, 9, 3] + [0.95, 0.88, 0.92, 0.91] \times [0.7, -0.3, 0.5, 0.8]$$

$$C_5 = [0.5, 0.35, 1.35, 0.24] + [0.665, -0.264, 0.46, 0.728]$$

$$C_5 = [1.165, 0.086, 1.81, 0.968]$$

(Krish context removed, friend context added!)

**Output Gate:**

$$o_5 = \sigma(W_o \cdot [h_4, x_5] + b_o) \approx [0.9, 0.85, 0.95, 0.88]$$

$$h_5 = [0.9, 0.85, 0.95, 0.88] \times \tanh([1.165, 0.086, 1.81, 0.968])$$

$$h_5 \approx [0.738, 0.072, 0.903, 0.66]$$

### Complete LSTM Architecture Summary

**All Gates Together:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \times \tanh(C_t)$$

**Information Flow:**

```
Previous state (h_{t-1}, C_{t-1}) + Current input (x_t)
                    ↓
        ┌───────────┴───────────┐
        ↓           ↓           ↓
    [Forget]   [Input]     [Output]
        ↓           ↓           ↓
    Remove old  Add new    Control output
        ↓___________↓___________|
                    ↓
        New state (h_t, C_t)
```

## ❓ Interview Questions & Answers

**Q1: What is the memory cell in LSTM?**

Memory cell ($C_t$) is the long-term memory component in LSTM, like a conveyor belt where:
- Information flows through time
- Gates can **add** new information (input gate)
- Gates can **remove** old information (forget gate)
- Solves vanishing gradient problem

**Q2: Explain the forget gate with an example.**

Forget gate decides what to forget:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Example: "Krish likes pizza but his friend likes burger"
- At "friend": Context switches
- Forget gate: $f_t \approx [0.1, 0.05, ...]$ (values near 0)
- Result: $C_t = f_t \times C_{t-1} \approx 0$ (Krish context forgotten)

**Q3: What does the input gate do?**

Input gate adds new information in two steps:

1. **Input gate (sigmoid)**: Decides which values to update

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

2. **Candidate values (tanh)**: Creates new values to add

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Combined: $i_t \times \tilde{C}_t$ = new information to add

**Q4: Why does LSTM use both sigmoid and tanh?**

- **Sigmoid** (0 to 1): Acts as gate (0=block, 1=pass)
  - Used in forget, input, output gates
  - Controls information flow

- **Tanh** (-1 to 1): Creates values/normalizes
  - Used in candidate values
  - Used to normalize cell state
  - Prevents value explosion

**Q5: What is context switching and how does LSTM handle it?**

Context switching = Sentence meaning changes

Example: "Krish likes pizza **but his friend** likes burger"

LSTM handles via:
1. **Forget gate**: Removes "Krish" context
2. **Input gate**: Adds "friend" context
3. **Cell state update**: $C_t = \text{forget old} + \text{add new}$

**Q6: What is the output gate's purpose?**

Output gate controls what information to send to next cell:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \times \tanh(C_t)$$

Combines:
- Normalized cell state: $\tanh(C_t)$
- Output gate: Decides which parts to output

**Q7: How is cell state updated in LSTM?**

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

Two operations:
1. **Multiplication**: $f_t \times C_{t-1}$ (remove old via forget gate)
2. **Addition**: $+ i_t \times \tilde{C}_t$ (add new via input gate)

This additive structure prevents vanishing gradient!

**Q8: What happens when there's no context switch?**

Example: "Krish likes pizza but he does not like burger"

- "he" refers to Krish (no switch)
- Forget gate: $f_t \approx [0.9, 0.95, ...]$ (keep context)
- Result: $C_t = f_t \times C_{t-1}$ retains most information

**Q9: Why does LSTM not have vanishing gradient?**

Because of **additive** cell state update:

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

Gradient flows through **addition** (+), not multiplication (×):
- Addition preserves gradients
- Direct path through cell state
- Gates control flow, not suppress it

**Q10: How many weight matrices does LSTM have?**

**8 weight matrices** (4 gates × 2 each):

Forget gate: $W_f$, $b_f$
Input gate: $W_i$, $b_i$, $W_C$, $b_C$
Output gate: $W_o$, $b_o$

Each gate operates on $[h_{t-1}, x_t]$ concatenation.

**Q11: What is the conveyor belt analogy for memory cell?**

Airport luggage conveyor belt:
- Luggage continuously moves (information flows)
- Can add luggage (input gate adds info)
- Can remove luggage (forget gate removes info)
- Belt keeps moving (cell state persists)

Memory cell works same way with information!

**Q12: Why use tanh on cell state before output?**

$$h_t = o_t \times \tanh(C_t)$$

**Reason**: Normalize cell state
- Cell state can grow large over time
- Tanh bounds values to [-1, 1]
- Prevents exploding values
- Makes output gate effective

## 💡 Key Takeaways

- **Memory Cell** ($C_t$) = Conveyor belt (add/remove info)
- **Forget Gate**: Removes old context (values near 0 = forget)
- **Input Gate**: Adds new context (sigmoid + tanh)
- **Output Gate**: Controls output (combines with normalized cell state)
- **Cell State Update**: $C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$
- **Context Switching**: Forget gate removes old, input gate adds new
- **No Vanishing Gradient**: Additive update preserves gradients
- **Tanh Normalization**: Prevents cell state explosion
- **8 Weight Matrices**: 4 gates, each with weights and bias
- **Visual Notation**: σ = sigmoid, tanh = tanh, × = multiply, + = add

## ⚠️ Common Mistakes

**Mistake 1**: "Forget gate forgets entire cell state"
- **Reality**: Outputs 0-1 per dimension, can partially forget

**Mistake 2**: "Input gate only uses sigmoid"
- **Reality**: Uses both sigmoid (selector) and tanh (values)

**Mistake 3**: "Output gate is the final output"
- **Reality**: Output gate creates hidden state, which is used for prediction

**Mistake 4**: "Cell state and hidden state are same"
- **Reality**: $C_t$ = long-term memory, $h_t$ = output/short-term

**Mistake 5**: "LSTM always forgets on context switch"
- **Reality**: Trained to recognize context switches, not automatic

**Mistake 6**: "Memory cell only adds information"
- **Reality**: Both adds (input gate) and removes (forget gate)

## 📝 Quick Revision Points

### Three Gates in LSTM

**Forget Gate:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Purpose: Remove old context
Output: 0 (forget) to 1 (keep)

**Input Gate:**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Purpose: Add new context
Two parts: selector (sigmoid) + values (tanh)

**Output Gate:**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \times \tanh(C_t)$$

Purpose: Control output
Combines normalized cell state with gate

### Cell State Update

$$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$$

$$C_t = \text{(Remove old)} + \text{(Add new)}$$

### Visual Notation

| Symbol | Meaning |
|--------|---------|
| σ | Neural network with sigmoid activation |
| tanh | Neural network with tanh activation |
| ⊗ or × | Pointwise multiplication |
| ⊕ or + | Pointwise addition |
| Lines joining | Concatenation |
| Lines splitting | Copy |

### Context Switching Example

```
"Krish likes pizza but his friend likes burger"

At "friend":
  Forget gate: f_t ≈ 0   → Forget "Krish"
  Input gate: i_t ≈ 1    → Add "friend"
  Result: Context switched successfully
```

### Remember

- **Memory cell** = Information highway
- **Forget gate** = Remove old context
- **Input gate** = Add new context (sigmoid + tanh)
- **Output gate** = Control output
- **Context switch** = Forget old + add new
- **No vanishing** = Additive cell state update
- **Tanh normalization** = Prevent explosion
- **8 weight matrices** = 4 gates × 2
