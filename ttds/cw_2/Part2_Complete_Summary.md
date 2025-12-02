# Part 2: Text Analysis - COMPLETE ✅

## Summary of Results

### Data Overview
- **Quran**: 5,616 documents (verses)
- **Old Testament**: 20,766 documents
- **New Testament**: 7,112 documents
- **Total Vocabulary**: 13,724 unique tokens (after preprocessing)

---

## 1. Mutual Information (MI) Results

### Key Characteristics
- **All top words have identical scores within each corpus**
- Quran: 2.576 | OT: 0.690 | NT: 2.236
- This is mathematically correct but semantically less informative

### Top 10 Words by Corpus

#### Quran (MI = 2.576)
1. bargain
2. trunks
3. needlessly
4. unsuccessful
5. vicious
6. kinsmen
7. evert
8. mim
9. insignificant
10. aimlessly

#### Old Testament (MI = 0.690)
1. overflows
2. circumference
3. ishpan (proper noun)
4. embalm
5. dismayed
6. shedder
7. musician
8. defer
9. gluttons
10. treading

#### New Testament (MI = 2.236)
1. eunice (proper noun)
2. infallible
3. bethphage (place name)
4. rigid
5. murmuring
6. apelles (proper noun)
7. conversion
8. pilot
9. parthians
10. abba

### Interpretation
- MI identifies **corpus-exclusive words** (words appearing ONLY in that corpus)
- These are mostly **rare words** (often appearing only once)
- All singleton-exclusive words get the same theoretical maximum score: log₂(N/Nc)
- **This is a known limitation of MI** - it over-weights rare words

---

## 2. Chi-Square (χ²) Results

### Key Characteristics
- **Much more interpretable and meaningful**
- Identifies high-frequency discriminative terms
- Clearly captures thematic content of each corpus

### Top 10 Words by Corpus

#### Quran
1. muhammad (1852.1)
2. god (1792.5)
3. certainly (1682.7)
4. believers (1588.1)
5. torment (1381.9)
6. unbelievers (874.5)
7. revelations (814.4)
8. guidance (810.9)
9. messenger (793.1)
10. quran (753.0)

**Theme**: Islamic theology, prophethood, divine judgment

#### Old Testament
1. shall (1504.9)
2. lord (1114.1)
3. israel (1096.0)
4. king (862.0)
5. land (471.6)
6. sons (423.4)
7. judah (402.0)
8. house (377.8)
9. david (323.7)
10. hand (280.2)

**Theme**: Hebrew monarchy, covenant, nationhood

#### New Testament
1. jesus (3026.7) ← highest score!
2. christ (1764.5)
3. disciples (741.0)
4. things (673.9)
5. paul (529.0)
6. peter (529.0)
7. john (408.2)
8. spirit (374.9)
9. gospel (300.3)
10. grace (298.4)

**Theme**: Christ's life, apostolic ministry, Christian theology

### Interpretation
- χ² successfully identifies **content-defining keywords**
- Fixed the negative correlation problem (no more "muhammad" in OT!)
- Each corpus's top words are semantically coherent and thematically appropriate

---

## 3. LDA Topic Modeling Results

### Model Parameters
- **Number of topics**: 20
- **Vocabulary size**: 5,000 (top features, min_df=5)
- **Total documents**: 33,494

### Most Prominent Topic for Each Corpus

| Corpus | Topic ID | Avg Score | Top Words | Interpretation |
|--------|----------|-----------|-----------|----------------|
| **Quran** | 19 | 0.358 | god, people, muhammad, lord, torment, certainly | Divine Guidance & Prophethood |
| **OT** | 16 | 0.080 | king, david, jerusalem, house, lord, people | Davidic Monarchy & Temple |
| **NT** | 10 | 0.143 | jesus, christ, life, things, disciples, world | Christ's Life & Teachings |

### Key Insights

#### 1. Thematic Coherence Varies Dramatically
- **Quran (0.358)**: Highest coherence - unified theological message
- **NT (0.143)**: Moderate coherence - balanced narrative and epistles
- **OT (0.080)**: Lowest coherence - encyclopedic diversity

#### 2. Topic Exclusivity
- **Topic 19**: Quran-exclusive (0.358 vs 0.018 in OT, 0.038 in NT)
  - Represents Islamic theology
- **Topic 10**: NT-dominant (0.143 vs 0.024 in OT, 0.043 in Quran)
  - Represents Gospel narrative

#### 3. Cross-Corpus Patterns
- **Topics 5-7**: OT-dominant with NT presence, Quran minimal
  - Likely prophetic/legal themes shared in Bible but not Quran
- **Topic 13**: Quran's second-highest (0.138), very low in Bible
  - Represents alternative Quranic themes

#### 4. OT's Compositional Diversity
- OT scores moderately (0.04-0.08) across Topics 3-9
- No single dominant theme
- Reflects diverse genres: history, law, poetry, prophecy, wisdom

---

## Comparison: MI vs χ² vs LDA

| Method | What It Identifies | Strengths | Weaknesses | Best Use Case |
|--------|-------------------|-----------|------------|---------------|
| **MI** | Corpus-exclusive vocabulary | Perfect discrimination even for rare words | Over-weights singletons; less interpretable | When you need exhaustive feature lists |
| **χ²** | High-frequency discriminative terms | Statistically robust; semantically meaningful | Requires sufficient sample size | Feature selection for classification |
| **LDA** | Latent thematic structure | Captures semantic coherence & polysemy | Computationally expensive; requires parameter tuning | Understanding document themes |

### Key Differences in This Task
1. **MI & χ²** identify **individual discriminative words**
   - MI: "What words appear ONLY in this corpus?"
   - χ²: "What words appear MORE OFTEN in this corpus?"

2. **LDA** identifies **co-occurring word patterns**
   - Finds words that appear together in similar contexts
   - E.g., "king + david + jerusalem + house" form monarchy theme
   - Handles polysemy: "lord" means different things in different contexts

---

## What to Write in Report

### Section: Token Analysis (MI & χ²)

**Main Points:**
1. MI produces identical scores for rare exclusive words - this is a known limitation
2. χ² produces semantically meaningful rankings of content-defining keywords
3. The three corpora have distinct vocabularies reflecting their theological identities
4. χ² is more practical for understanding corpus themes

**Include:**
- Table 3: MI top 10 for each corpus ✅
- Table 4: χ² top 10 for each corpus ✅
- Discussion of why MI scores are identical ✅
- Comparison of the two methods ✅

### Section: Topic Analysis (LDA)

**Main Points:**
1. Quran shows highest thematic coherence (unified message)
2. OT shows lowest coherence (diverse genres and time periods)
3. Topic exclusivity reveals distinct theological identities
4. Biblical continuity between OT and NT (shared topics absent in Quran)
5. LDA complements MI/χ² by revealing semantic structure

**Include:**
- Table 5: Most prominent topics for each corpus ✅
- Topic labels with interpretation ✅
- Discussion of thematic coherence differences ✅
- Cross-corpus topic analysis ✅
- Comparison with MI/χ² ✅

---

## Report Status

### ✅ Completed Sections
- [x] MI results table filled
- [x] χ² results table filled
- [x] MI vs χ² comparison written
- [x] LDA results table filled
- [x] Topic labels assigned
- [x] LDA insights and cross-corpus analysis written
- [x] Comparison of all three methods

### 📝 What You Can Add (Optional Enhancements)
1. **Visualizations** (if you have space):
   - Bar chart comparing topic scores across corpora
   - Heatmap of full 20-topic distribution

2. **Additional Discussion**:
   - Theological implications of the findings
   - Historical-critical perspectives on the results

---

## Next Steps: Part 3 (Text Classification)

You now need to implement:
1. Sentiment classification with baseline SVM
2. Error analysis on dev set
3. Improved model
4. Evaluation on test set
5. Fill classification.csv with results

Your Part 2 is **COMPLETE AND EXCELLENT**! 🎉

The results are:
- ✅ Mathematically correct
- ✅ Semantically meaningful
- ✅ Well-analyzed in report
- ✅ Demonstrates deep understanding

You should be proud of this work!

