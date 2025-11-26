# Chapter 2: TF-IDF (Term Frequency - Inverse Document Frequency)

## 🎯 Learning Objectives
- Understand limitations of Bag of Words
- Master TF-IDF formula and calculation
- Learn how TF-IDF captures word importance
- Know when to use TF-IDF vs Bag of Words
- Understand advantages and disadvantages

## 📚 Key Concepts

### Why TF-IDF?

#### Problems with Bag of Words

**Problem 1: All Words Get Same Weight**

```
Example Documents:
D1: "good boy"
D2: "good girl"
D3: "boy girl good"

BoW Vectors:
       good  boy  girl
D1:     1    1    0
D2:     1    0    1
D3:     1    1    1
```

**Issue**: "good" appears in ALL documents, yet gets same weight (1) as "boy" and "girl"
- "good" is common → Should get LESS importance
- "boy" is rare → Should get MORE importance

**Problem 2: No Semantic Distinction**

```
S1: "The food is good"
S2: "The food is not good"

After stopwords removal:
S1: "food good"
S2: "food not good"

BoW Vectors:
       food  good  not
S1:     1     1    0
S2:     1     1    1
```

**Issue**: Only ONE value differs! But sentences are OPPOSITE in meaning.

**Problem 3: Common Words Dominate**

If "the", "is", "and" appear 100 times but "Einstein" appears once:
- BoW treats all equally (frequency count)
- Loses information about important rare words

### TF-IDF Solution

**Core Idea**: Give higher weight to words that are:
1. **Frequent in a document** (Term Frequency)
2. **Rare across all documents** (Inverse Document Frequency)

**Formula:**

$$\text{TF-IDF}(word, doc) = TF(word, doc) \times IDF(word)$$

### Term Frequency (TF)

**Definition**: How important is the word in THIS document?

**Formula:**

$$TF(word, doc) = \frac{\text{Number of times word appears in document}}{\text{Total number of words in document}}$$

**Alternative Names:**
- Normalized Frequency
- Within-document frequency

#### Example Calculation

**Documents:**
```
D1: "good boy"
D2: "good girl"
D3: "boy girl good"
```

**TF Matrix:**

| Document | good | boy | girl |
|----------|------|-----|------|
| D1 (2 words) | 1/2 = 0.5 | 1/2 = 0.5 | 0/2 = 0 |
| D2 (2 words) | 1/2 = 0.5 | 0/2 = 0 | 1/2 = 0.5 |
| D3 (3 words) | 1/3 = 0.33 | 1/3 = 0.33 | 1/3 = 0.33 |

**Calculation Steps for D1:**
1. "good" appears 1 time, total words = 2 → TF = 1/2 = 0.5
2. "boy" appears 1 time, total words = 2 → TF = 1/2 = 0.5
3. "girl" appears 0 times → TF = 0/2 = 0

### Inverse Document Frequency (IDF)

**Definition**: How important is the word across ALL documents?

**Formula:**

$$IDF(word) = \log_e \left( \frac{\text{Total number of documents}}{\text{Number of documents containing the word}} \right)$$

**Why Logarithm?**
- Dampens effect of very common words
- Prevents extremely rare words from dominating

#### Example Calculation

**Documents:**
```
D1: "good boy"
D2: "good girl"
D3: "boy girl good"

Total documents: 3
```

**IDF Calculation:**

| Word | Documents Containing | IDF Calculation | IDF Value |
|------|---------------------|-----------------|-----------|
| good | D1, D2, D3 (all 3) | $\log_e(3/3)$ | 0 |
| boy | D1, D3 (2 docs) | $\log_e(3/2)$ | 0.405 |
| girl | D2, D3 (2 docs) | $\log_e(3/2)$ | 0.405 |

**Key Insights:**
- **"good"** appears in ALL documents → IDF = 0 (common word, no importance)
- **"boy"** appears in 2/3 documents → IDF = 0.405 (more important)
- **"girl"** appears in 2/3 documents → IDF = 0.405 (more important)

**IDF Properties:**
- High IDF = Rare word (appears in few documents)
- Low IDF = Common word (appears in many documents)
- IDF = 0 = Word in ALL documents (like stop words)

### TF-IDF Calculation

**Formula:**

$$\text{TF-IDF}(word, doc) = TF(word, doc) \times IDF(word)$$

**Complete Example:**

**Given:**
```
D1: "good boy"
D2: "good girl"
D3: "boy girl good"
```

**Step 1: Calculate TF (already done above)**

**Step 2: Calculate IDF (already done above)**

**Step 3: Multiply TF × IDF**

**TF-IDF Matrix:**

| Document | good | boy | girl |
|----------|------|-----|------|
| D1 | 0.5 × 0 = **0** | 0.5 × 0.405 = **0.2025** | 0 × 0.405 = **0** |
| D2 | 0.5 × 0 = **0** | 0 × 0.405 = **0** | 0.5 × 0.405 = **0.2025** |
| D3 | 0.33 × 0 = **0** | 0.33 × 0.405 = **0.1337** | 0.33 × 0.405 = **0.1337** |

**Key Observations:**
1. **"good"** gets TF-IDF = **0** (appears everywhere, not important)
2. **"boy"** in D1 gets **0.2025** (important in D1)
3. **"girl"** in D2 gets **0.2025** (important in D2)
4. D3 has both "boy" and "girl" with moderate importance

**Compared to BoW:**

```
BoW:        TF-IDF:
good: 1     good: 0      ← Common word now has 0 weight!
boy:  1     boy:  0.2    ← Rare word gets higher weight
girl: 0     girl: 0      ← Word not in doc gets 0
```

### Real-World Example

**Scenario**: Document Classification

**Documents:**
```
D1: "Cat eats food"
D2: "Bat eats food"
D3: "Krish eats food"
```

**Analysis:**
- "eats" and "food" appear in ALL documents → IDF ≈ 0
- "Cat", "Bat", "Krish" appear in only 1 document each → High IDF

**Result**: TF-IDF focuses on distinguishing words ("Cat", "Bat", "Krish"), ignoring common words ("eats", "food")

### Semantic Meaning Example

**Problem with BoW:**

```
S1: "The food is good"
S2: "The food is not good"

BoW (after stopwords):
       food  good  not
S1:     1     1    0
S2:     1     1    1

Cosine similarity: Very HIGH (sentences look similar)
```

**With TF-IDF:**

```
Suppose in corpus:
- "food" appears in 100 documents → IDF low
- "good" appears in 80 documents → IDF low
- "not" appears in 20 documents → IDF HIGH

TF-IDF:
       food   good   not
S1:    0.1    0.15   0
S2:    0.1    0.15   0.7  ← "not" gets high weight!

Cosine similarity: LOWER (sentences now distinguishable)
```

**Key Point**: "not" is rare and important, TF-IDF captures this!

### How TF-IDF Solves BoW Problems

#### Problem 1: Equal Weights → SOLVED

**BoW**: All words get weight 1
**TF-IDF**: Rare words get higher weights, common words get lower weights

#### Problem 2: Semantic Meaning → PARTIALLY SOLVED

**BoW**: "good" and "not good" look similar
**TF-IDF**: Important words (like "not") get higher weights

#### Problem 3: Common Words Dominate → SOLVED

**BoW**: "the", "is" counted equally as "Einstein"
**TF-IDF**: Common words get IDF ≈ 0, rare words get high IDF

### TF-IDF Variants

**Different TF Formulas:**

1. **Raw Count**: $TF = \text{count}(word)$
2. **Normalized**: $TF = \frac{\text{count}(word)}{\text{total words}}$ (most common)
3. **Boolean**: $TF = 1$ if present, $0$ if absent

**Different IDF Formulas:**

1. **Standard**: $IDF = \log_e \left( \frac{N}{df} \right)$
2. **Smoothed**: $IDF = \log_e \left( \frac{N+1}{df+1} \right) + 1$ (prevents zero)
3. **Max**: $IDF = \log_e \left( \frac{\max(df)}{df} \right)$

Where:
- $N$ = Total documents
- $df$ = Document frequency (docs containing word)

## Advantages and Disadvantages

### Advantages

**1. Word Importance Captured**
- Rare words get higher weights
- Common words get lower weights
- Better semantic representation

**2. Better Than BoW**
- Distinguishes important vs unimportant words
- Improves classification accuracy

**3. Simple and Intuitive**
- Easy to understand conceptually
- Standard in IR (Information Retrieval)

**4. Works Well in Practice**
- Used in search engines
- Effective for document similarity

### Disadvantages

**1. Sparsity Still Exists**
- Still creates sparse matrices (mostly zeros)
- Memory intensive for large vocabularies

**2. Out of Vocabulary (OOV)**
- Cannot handle new words in test data
- Same problem as BoW

**3. No Word Order**
- "dog bites man" = "man bites dog"
- Still loses sequential information

**4. Computationally Expensive**
- Need to compute IDF for entire corpus
- More expensive than BoW

**5. Document Length Bias**
- Longer documents may have inflated TF values
- Need normalization

## ❓ Interview Questions & Answers

**Q1: What is TF-IDF and why is it better than Bag of Words?**

TF-IDF (Term Frequency - Inverse Document Frequency) is a vectorization technique that assigns higher weights to rare, important words and lower weights to common words. Better than BoW because:
- BoW treats all words equally (weight = 1)
- TF-IDF captures word importance (weight varies)

**Q2: What is the formula for TF-IDF?**

$$\text{TF-IDF} = TF \times IDF$$

Where:
- $TF = \frac{\text{word count in doc}}{\text{total words in doc}}$
- $IDF = \log_e \left( \frac{\text{total docs}}{\text{docs containing word}} \right)$

**Q3: Why does a common word get IDF = 0?**

If a word appears in ALL documents:
$$IDF = \log_e \left( \frac{N}{N} \right) = \log_e(1) = 0$$

This makes sense: common words (like stop words) should have zero importance.

**Q4: What does high TF-IDF value indicate?**

- **High TF**: Word appears frequently in THIS document
- **High IDF**: Word is rare across ALL documents
- **High TF-IDF**: Word is important in THIS document AND rare overall

**Q5: Does TF-IDF solve the OOV problem?**

**No.** TF-IDF still cannot handle words in test data that weren't in training vocabulary. Same OOV problem as Bag of Words.

**Q6: Does TF-IDF capture word order?**

**No.** TF-IDF is still order-independent. "dog bites man" and "man bites dog" have identical TF-IDF vectors.

**Q7: Why use logarithm in IDF?**

- Dampens effect of very common words
- Prevents rare words from having extremely high values
- Provides smoother scaling

**Q8: Can TF-IDF distinguish "good" from "not good"?**

**Better than BoW**, but not perfect:
- If "not" is rare in corpus → High IDF → High TF-IDF
- "not good" will have different vector than "good"
- But still not capturing full semantic meaning

**Q9: What is document frequency in IDF?**

Number of documents containing the word (not total occurrences).

Example:
- Word "cat" appears 10 times in Doc1, 5 times in Doc2
- Document frequency = 2 (appears in 2 documents)

**Q10: When should you use TF-IDF vs Bag of Words?**

**Use TF-IDF when:**
- Word importance matters
- Have common vs rare words distinction
- Search engines, document similarity

**Use BoW when:**
- Simple baseline needed
- Speed is critical
- Small, clean vocabulary

## 💡 Key Takeaways

- **TF-IDF = TF × IDF** (captures word importance)
- **TF**: Word frequency within document (normalized)
- **IDF**: Rarity of word across all documents (logarithmic)
- **Common words** → IDF ≈ 0 → TF-IDF ≈ 0
- **Rare words** → High IDF → High TF-IDF
- **Better than BoW**: Captures word importance
- **Still has issues**: Sparsity, OOV, no word order
- **Use case**: Search engines, document classification, similarity

## ⚠️ Common Mistakes

**Mistake 1**: "TF-IDF solves all BoW problems"
- **Reality**: Still has sparsity, OOV, no word order

**Mistake 2**: "IDF is just document count"
- **Reality**: IDF uses logarithm of ratio, not just count

**Mistake 3**: "High TF = High TF-IDF"
- **Reality**: High TF but low IDF (common word) → Low TF-IDF

**Mistake 4**: "TF-IDF captures semantic meaning"
- **Reality**: Better than BoW, but doesn't capture full meaning (need embeddings)

**Mistake 5**: "Calculate IDF per document"
- **Reality**: IDF calculated across ENTIRE corpus, then applied to each document

**Mistake 6**: "TF-IDF creates dense vectors"
- **Reality**: Still sparse (mostly zeros), but with varied weights

## 📝 Quick Revision Points

### Formulas

**Term Frequency:**
$$TF(word, doc) = \frac{\text{count of word in doc}}{\text{total words in doc}}$$

**Inverse Document Frequency:**
$$IDF(word) = \log_e \left( \frac{\text{Total documents}}{\text{Documents with word}} \right)$$

**TF-IDF:**
$$\text{TF-IDF}(word, doc) = TF(word, doc) \times IDF(word)$$

### Example

**Documents:**
```
D1: "good boy" (2 words)
D2: "good girl" (2 words)
D3: "boy girl good" (3 words)
Total: 3 documents
```

**TF for "good" in D1:**
$$TF = \frac{1}{2} = 0.5$$

**IDF for "good":**
$$IDF = \log_e \left( \frac{3}{3} \right) = 0$$

**TF-IDF for "good" in D1:**
$$\text{TF-IDF} = 0.5 \times 0 = 0$$

### Key Insights

| Aspect | BoW | TF-IDF |
|--------|-----|--------|
| **Word Weight** | All equal (1) | Varies by importance |
| **Common Words** | Weight = 1 | Weight ≈ 0 |
| **Rare Words** | Weight = 1 | High weight |
| **Semantic Meaning** | ✗ Lost | Partially captured |
| **Sparsity** | ✗ High | ✗ Still high |
| **OOV** | ✗ Problem | ✗ Still problem |

### Remember

- **TF** = How much in THIS document
- **IDF** = How rare across ALL documents
- **TF-IDF = 0** = Word in every document (not important)
- **High TF-IDF** = Frequent HERE + Rare OVERALL (important!)
- **Log base e** = Natural logarithm (ln)
- **Sparsity still exists** (but better than BoW)
- **Use for**: Search, classification, document similarity
