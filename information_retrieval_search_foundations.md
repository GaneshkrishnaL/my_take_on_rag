


## Metadata Filtering

### 💡 One-Liner

> Metadata filtering = SQL WHERE clause for your retriever.
> It narrows documents by **rigid criteria on tags** (not content), before other search techniques kick in.

---

### 🔍 What Is It?

Metadata filtering uses **strict, exact-match conditions** on document attributes to decide which documents even enter the search pipeline. Think of it as a bouncer at the door — it doesn't care what's inside the document, only what's on the label.

```
┌──────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE BASE                            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Article  │  │ Article  │  │ Article  │  │ Article  │  ...    │
│  │──────────│  │──────────│  │──────────│  │──────────│        │
│  │title: ...│  │title: ...│  │title: ...│  │title: ...│        │
│  │date: ... │  │date: ... │  │date: ... │  │date: ... │        │
│  │author:...│  │author:...│  │author:...│  │author:...│        │
│  │section:..│  │section:..│  │section:..│  │section:..│        │
│  │access:   │  │access:   │  │access:   │  │access:   │        │
│  │ free/paid│  │ free/paid│  │ free/paid│  │ free/paid│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│         ▼ Apply Filter: section="opinion" AND date=Jun-Jul 2024  │
│                                                                  │
│  ┌──────────┐  ┌──────────┐                                     │
│  │ Match ✅ │  │ Match ✅ │   ← Only these pass through          │
│  └──────────┘  └──────────┘                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

### 🗞️ Newspaper Example — Step by Step

Imagine a newspaper archive with thousands of articles. Each article has metadata tags:

| Metadata Field | Example Value         |
|----------------|----------------------|
| `title`        | "Election 2024 Recap"|
| `date`         | 2024-07-15           |
| `author`       | "Jane Doe"           |
| `section`      | "Opinion"            |
| `access`       | "paid"               |
| `region`       | "US-East"            |

**Simple query** — one filter:

```
WHERE author = "Jane Doe"
→ Returns: every article Jane ever wrote
```

**Compound query** — multiple filters stacked:

```
WHERE section = "opinion"
  AND date BETWEEN "2024-06-01" AND "2024-07-31"
  AND author = "Jane Doe"
→ Returns: only Jane's opinion pieces from summer 2024
```

> 🧠 **Analogy:** If you've ever filtered a spreadsheet by column values, you've already done metadata filtering.

---

### 🏗️ Where Does It Fit in a RAG Pipeline?

Metadata filtering is **NOT** the main retriever. It's the **pre-filter** that runs before semantic or keyword search to shrink the candidate pool.

```
User Query
    │
    ▼
┌───────────────────────┐
│  METADATA FILTER      │  ← Rigid rules (access, region, date, etc.)
│  (Pre-filter step)    │
└───────────┬───────────┘
            │  Smaller candidate set
            ▼
┌───────────────────────┐
│  KEYWORD / SEMANTIC   │  ← Actual content-based search
│  SEARCH               │
└───────────┬───────────┘
            │  Ranked results
            ▼
┌───────────────────────┐
│  LLM GENERATION       │
└───────────────────────┘
```

**Key insight:** The filters are usually driven by **user attributes** (who is asking), not the query text (what they're asking).

---

### 🔐 Real-World Use Cases

| Scenario | Metadata Used | How It Works |
|----------|--------------|--------------|
| **Paywall enforcement** | `access: free / paid` | System checks if user is a subscriber → if not, filter out `paid` articles |
| **Regional content** | `region: US-East` | Detect user's location → only return articles from their region |
| **Time-scoped search** | `date` | User asks about "recent news" → filter to last 30 days |
| **Role-based access** | `clearance_level` | Enterprise doc search → only show docs the user's role can access |

---

### ✅ Pros vs ❌ Cons

```
 ✅ STRENGTHS                          ❌ LIMITATIONS
 ─────────────────────────────         ─────────────────────────────
 • Simple to understand & debug        • NOT a search technique by itself
 • Fast & well-optimized               • Ignores document CONTENT entirely
 • ONLY way to enforce rigid           • Cannot RANK documents
   include/exclude rules               • Useless if used alone
 • Mature technology                   • Needs pairing with keyword or
                                         semantic search
```

---

### 🧩 Key Takeaways

1. **Metadata filtering = pre-filter, not search.** It narrows the pool; other techniques do the actual retrieval.
2. **Filters are driven by USER context** (subscription status, region, role), not the query itself.
3. **It's the ONLY way to enforce hard rules** like "never show paid content to free users."
4. **Always pair it** with keyword search, semantic search, or both.

---

## Keyword Search & TF-IDF

### 💡 One-Liner

> Keyword search = "do the document and the query share the same words?"
> Rank by how often those shared words appear, weighted by how rare they are → that's TF-IDF.

---

### 🔍 Core Idea — Bag of Words

Keyword search **throws away word order** completely. It only cares about **which words** appear and **how many times**.

```
Text: "making pizza without a pizza oven"

           making  pizza  without  a  oven  ...  (tens of thousands more)
Vector:  [  1       2       1      1    1   ...         0              ]

Most slots are 0 → that's why it's called a SPARSE VECTOR
```

> 🧠 **Analogy:** Imagine dumping all the words of a sentence into a bag, shaking it up, and just counting what's inside. That's a bag of words.

---

### 🗂️ Building the Index (Before Any Search Happens)

Every document in the knowledge base gets its own sparse vector. Stack them together and you get a **Term-Document Matrix**:

```
                    Doc A    Doc B    Doc C    Doc D
                  ┌────────┬────────┬────────┬────────┐
   making         │   1    │   0    │   2    │   0    │
   pizza          │   3    │   0    │   1    │   0    │
   oven           │   2    │   0    │   0    │   1    │
   the            │   5    │   4    │   3    │   6    │
   recipe         │   1    │   2    │   0    │   0    │
   ...            │  ...   │  ...   │  ...   │  ...   │
                  └────────┴────────┴────────┴────────┘

Also called an INVERTED INDEX because:
  Normal thinking:   Document → "what words does it have?"
  Inverted thinking:  Word → "which documents contain me?"
```

This index is built **once** ahead of time, not at query time.

---

### 📊 Scoring — From Simple to TF-IDF (4 Levels)

The transcript walks through progressively better scoring. Here's how each level works:

```
LEVEL 1 — Binary Match
────────────────────────────────────────────────────
Rule: Does the doc contain the keyword? → +1 point

Prompt: "making pizza without a pizza oven" (5 unique keywords)

         making  pizza  without   a   oven   SCORE
Doc A:     ✅      ✅      ✅     ✅    ✅      5
Doc B:     ❌      ❌      ✅     ✅    ❌      2
Doc C:     ✅      ✅      ❌     ✅    ❌      3

Problem: Doesn't care if "pizza" appears 1 time or 100 times.
```

```
LEVEL 2 — Term Frequency (TF)
────────────────────────────────────────────────────
Rule: Award points = number of times keyword appears in doc

         making  pizza  without   a   oven   SCORE
Doc A:     1       3       1      5     2      12
Doc C:     2       1       0      3     0       6

Problem: Longer documents score higher just because they have more words.
```

```
LEVEL 3 — Normalized TF
────────────────────────────────────────────────────
Rule: Divide score by document length (total word count)

Doc A: 12 points / 200 words = 0.060
Doc C:  6 points /  50 words = 0.120  ← shorter doc ranks HIGHER now

Problem: "a" and "the" contribute as much as "pizza" and "oven."
```

```
LEVEL 4 — TF-IDF (the good stuff)
────────────────────────────────────────────────────
Rule: Weight each word by how RARE it is across all documents

Step 1: Calculate IDF for each word

  IDF(word) = log( total_docs / docs_containing_word )

  Example (100 docs in knowledge base):
  ┌──────────┬───────────────┬────────┬─────────────┐
  │  Word    │ Docs with it  │  DF    │ IDF = log() │
  ├──────────┼───────────────┼────────┼─────────────┤
  │  "the"   │     100       │  1.00  │    0.0      │  ← appears everywhere, worthless
  │  "a"     │      95       │  0.95  │    0.02     │
  │  "pizza" │       5       │  0.05  │    1.30     │  ← rare = valuable signal
  │  "oven"  │       3       │  0.03  │    1.52     │  ← even rarer = even more valuable
  └──────────┴───────────────┴────────┴─────────────┘

Step 2: Multiply each cell in the matrix by that word's IDF
Step 3: Score documents using the updated TF-IDF matrix

Result: Documents with rare keywords like "pizza" and "oven"
        score WAY higher than docs that just have "a" and "the."
```

---

### 🔄 Full Flow — How Keyword Search Works End to End

```
 ① BUILD (one-time, offline)
 ─────────────────────────────────────────────
   Documents → Tokenize → Count words → Build Term-Document Matrix
                                         → Multiply by IDF weights
                                         → Store as TF-IDF Matrix

 ② QUERY (at search time)
 ─────────────────────────────────────────────
   Prompt → Tokenize → Sparse vector for prompt
                          │
                          ▼
                 For each keyword in prompt:
                   → Look up its row in TF-IDF matrix
                   → Add each document's TF-IDF score
                          │
                          ▼
                 Rank documents by total score
                          │
                          ▼
                 Return top-K documents
```

---

### 🧠 Why Log in IDF?

Without log, a word appearing in 1 out of 100 docs gets IDF = 100, while a word in 50 docs gets IDF = 2. That 50x difference is way too aggressive — one rare word would completely dominate scoring. The log compresses this to a gentler curve so rare words still win, but don't obliterate everything else.

---

### 🔮 What Comes Next: BM-25

TF-IDF is the **foundational baseline**, but modern systems typically use **BM-25** — a refined version that adds smarter term frequency saturation and better document length normalization. (Covered in the next topic.)

---

### ✅ Pros vs ❌ Cons

```
 ✅ STRENGTHS                          ❌ LIMITATIONS
 ─────────────────────────────         ─────────────────────────────
 • Actually looks at document          • Ignores word ORDER and MEANING
   CONTENT (unlike metadata)             ("dog bites man" = "man bites dog")
 • Fast — sparse vectors are           • Can't handle synonyms
   cheap to store & search               ("car" won't match "automobile")
 • Well-understood, decades of         • Common words dilute signal
   optimization                          (IDF helps but doesn't eliminate)
 • Great for exact-term matching       • No understanding of CONTEXT
   (product names, codes, IDs)
```

---

### 🧩 Key Takeaways

1. **Bag of words** — order doesn't matter, only word counts.
2. **Sparse vectors** — mostly zeros, one slot per vocabulary word.
3. **Inverted index** — start from a word, find all documents containing it. Built once, queried many times.
4. **TF-IDF** — weight words by rarity (IDF) so "pizza" matters more than "the."
5. **Still limited** — no understanding of meaning, synonyms, or context. That's where semantic search comes in.

---

## 📝 Appendix — Quick Glossary

| Term | Meaning |
|------|---------|
| **Metadata** | Descriptive tags attached to a document (title, date, author, etc.) — not the content itself |
| **Pre-filter** | A step that reduces the candidate document set before the main search runs |
| **Rigid criteria** | Exact-match or range-based conditions (yes/no, not "sort of relevant") |
| **Retriever** | The component in a RAG system responsible for finding relevant documents |
| **Bag of Words** | Representation where only word presence/count matters, order is discarded |
| **Sparse Vector** | A vector with mostly zeros — one dimension per vocabulary word |
| **Term-Document Matrix** | Grid where rows = words, columns = documents, cells = word counts |
| **Inverted Index** | Data structure that maps each word → list of documents containing it |
| **TF (Term Frequency)** | How many times a word appears in a specific document |
| **IDF (Inverse Document Frequency)** | log(total docs / docs containing the word) — measures how rare a word is |
| **TF-IDF** | TF × IDF — scores that reward frequent use of rare words |
| **BM-25** | A refined version of TF-IDF with better length normalization and term saturation (next topic) |

---

*This document is designed to grow. Each new topic in Module 2 will be appended above the Appendix section.*
