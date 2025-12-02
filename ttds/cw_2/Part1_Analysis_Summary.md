# Part 1: IR Evaluation - Detailed Analysis

## Statistical Significance Test Results

### Summary Table

| Metric | Best System | Score | 2nd System | Score | p-value | Significant? |
|--------|-------------|-------|------------|-------|---------|--------------|
| P@10 | 3 | 0.410 | 5 | 0.410 | 1.000 | ❌ NO |
| R@50 | 2 | 0.867 | 1 | 0.834 | 0.703 | ❌ NO |
| R-Precision | 3 | 0.449 | 6 | 0.449 | 1.000 | ❌ NO |
| AP | 3 | 0.451 | 6 | 0.445 | 0.967 | ❌ NO |
| nDCG@10 | 3 | 0.420 | 6 | 0.400 | 0.883 | ❌ NO |
| nDCG@20 | 3 | 0.511 | 6 | 0.490 | 0.868 | ❌ NO |

## Key Findings

### 1. Overall Winner: System 3
- **Dominates in 5 out of 6 metrics**
- Only loses to System 2 in R@50
- Shows consistent high performance across different evaluation dimensions

### 2. No Statistical Significance
**Critical Finding**: Despite System 3's apparent superiority, **none of the differences are statistically significant** at α = 0.05 level.

### 3. Why No Significance?

#### A. Perfect Ties (p = 1.000)
Two cases show perfect or near-perfect ties:
- **P@10**: Systems 3, 5, and 6 all scored exactly 0.410
- **R-Precision**: Systems 3 and 6 both scored 0.449

When means are identical, t-test yields p = 1.000, indicating zero difference.

#### B. Small Sample Size (n = 10)
With only 10 queries:
- **Limited statistical power** to detect differences
- High standard error
- Large confidence intervals
- Difficult to distinguish signal from noise

Example calculation for R@50:
```
System 2 mean: 0.867
System 1 mean: 0.834
Difference: 0.033 (3.3 percentage points)
```

While this seems meaningful, the variance across 10 queries is too large to conclude this isn't due to random chance.

#### C. High Variance Across Queries
IR performance varies dramatically by query:
- Some queries are "easy" (many relevant docs)
- Some queries are "hard" (few relevant docs)
- Query difficulty dominates system differences
- 10 queries insufficient to average out this variance

#### D. Similar System Designs
Systems 3, 5, and 6 show remarkably similar performance patterns:
- Suggests they use similar retrieval algorithms
- Small implementation differences don't lead to statistically distinguishable performance
- All are likely variations of the same basic approach

## Detailed Metric Analysis

### Precision@10 (P@10)
- **Three-way tie**: Systems 3, 5, 6 (0.410)
- **Interpretation**: These systems correctly identify relevant docs in top 10 about 41% of the time
- **System 4 worst**: 0.080 (only 8% precision)

### Recall@50 (R@50)
- **Winner**: System 2 (0.867)
- **Interpretation**: System 2 finds 86.7% of all relevant docs within top 50 results
- **Trade-off**: System 2 sacrifices precision for recall (P@10 only 0.220)

### R-Precision
- **Two-way tie**: Systems 3, 6 (0.449)
- **Definition**: Precision at R, where R = number of relevant docs for that query
- **Adaptive metric**: Adjusts cutoff per query

### Average Precision (AP)
- **Winner**: System 3 (0.451)
- **Runner-up**: System 6 (0.445)
- **Very close**: Difference of only 0.006
- **Interpretation**: Captures both precision and ranking quality

### nDCG@10 and nDCG@20
- **Winner**: System 3 (0.420 and 0.511)
- **Runner-up**: System 6 (0.400 and 0.490)
- **Uses graded relevance**: Unlike binary metrics, nDCG rewards highly-relevant (rel=3) docs more
- **System 3's strength**: Better at ranking highly-relevant docs near top

## What This Means for Your Report

### ✅ What to Write

1. **Acknowledge the winner**: System 3 is consistently best
2. **Report the lack of significance**: Be honest about p-values
3. **Explain WHY**: This is crucial - don't just say "no significance"
4. **Provide context**: 
   - Small n problem
   - High variance
   - Tied scores
   - Similar systems

### ❌ What NOT to Write

1. Don't claim System 3 is "significantly better" (it's not, statistically)
2. Don't ignore the p-values
3. Don't dismiss the lack of significance without explanation
4. Don't pretend System 4's poor performance is acceptable

### 📝 Example Paragraph for Report

> "System 3 achieved the highest mean scores across five of six metrics (P@10=0.410, R-Precision=0.449, AP=0.451, nDCG@10=0.420, nDCG@20=0.511), suggesting superior overall retrieval quality. However, two-tailed t-tests revealed no statistically significant differences between the best and second-best systems for any metric (all p > 0.05). This lack of significance is primarily attributable to the limited sample size (n=10 queries), which provides insufficient statistical power to detect small performance differences. Additionally, several metrics showed tied performances (e.g., Systems 3, 5, and 6 all achieved P@10=0.410), yielding p-values of 1.000. Despite the absence of statistical significance, System 3's consistent top performance across multiple evaluation dimensions suggests it is the most reliable choice in practice."

## System Rankings Summary

### By Number of Metrics Won
1. **System 3**: 5 wins (best overall)
2. **System 2**: 1 win (R@50 specialist)
3. **System 6**: 0 wins but 4 second-places (close competitor)
4. **System 1**: Strong in R@50 only
5. **System 5**: Tied for P@10 but weaker elsewhere
6. **System 4**: Poor across all metrics

### Recommendation
**For practical deployment**: Choose **System 3**
- Most balanced performance
- Excels in both precision and ranking quality
- Consistent across different evaluation perspectives

**For recall-critical applications**: Consider **System 2**
- Best at finding relevant documents
- Trade-off: Lower precision in top results

## T-Test Interpretation Guide

### p-value Ranges
- **p < 0.001**: Highly significant (***) - Very strong evidence
- **p < 0.01**: Very significant (**) - Strong evidence
- **p < 0.05**: Significant (*) - Sufficient evidence
- **p ≥ 0.05**: Not significant (ns) - Insufficient evidence

### Your Results
All p-values are **> 0.70**, which means:
- No evidence of difference
- Observed differences likely due to chance
- Need more queries to detect real differences (if they exist)

### Effect Size vs. Statistical Significance
Important distinction:
- **Effect size**: How different the scores are (practical importance)
- **Statistical significance**: Whether difference is reliably non-zero

Your case:
- **Moderate effect sizes** (e.g., 0.451 vs 0.445 in AP)
- **No statistical significance** (insufficient data to confirm)

This is why you should discuss **both** in your report!



