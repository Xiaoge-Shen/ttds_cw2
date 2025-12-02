# TTDS Coursework 2 Report Guide

## LaTeX Template Structure

The `Report.tex` file has been created with the following structure:

### Section 1: Implementation Overview (1 page)
- Code structure explanation
- Key implementation challenges
- What was learned
- **Status**: ✅ Complete - includes detailed analysis of your IR implementation

### Section 2: IR Evaluation Results (1 page)
- **Status**: ✅ Complete with your actual results
- Table 1: Mean performance of all 6 systems across all metrics
- Table 2: Statistical significance analysis
- Detailed explanation of why NO systems showed statistical significance

#### Key Findings Already Included:
- System 3 is the best overall performer (highest in 5/6 metrics)
- NO statistical significance found (all p-values > 0.05)
- Explained reasons:
  - Small sample size (10 queries)
  - Tied performances
  - High variance across queries
  - Similar retrieval strategies

### Section 3: Text Analysis (1-2 pages)
- **Status**: 🔨 TODO - Awaiting your Part 2 implementation
- Placeholder tables for MI and Chi-square top 10 tokens
- Placeholder table for LDA topic analysis
- Discussion sections for comparing MI vs Chi-square vs LDA

### Section 4: Text Classification (1-2 pages)
- **Status**: 🔨 TODO - Awaiting your Part 3 implementation
- Baseline system description
- Error analysis section (3 examples)
- Improved system description
- Performance comparison tables

## How to Fill in the Template

### For Part 2 (Text Analysis):
1. Run your MI and Chi-square analysis
2. Replace the "TODO" sections in Tables 3 and 4 with your top 10 tokens
3. Fill in the discussion sections about:
   - Differences between MI and Chi-square
   - What you learned about each corpus
4. Run LDA and fill in Table 5 with your top topics
5. Provide topic labels and discuss common/different themes

### For Part 3 (Classification):
1. Run baseline system and fill in Table 6
2. Add 3 misclassified examples with hypotheses
3. Describe your improvements in detail
4. Fill in Table 7 with improved results
5. Calculate and report F1-macro gains

## Compiling the LaTeX Document

```bash
# Standard compilation
pdflatex Report.tex
pdflatex Report.tex  # Run twice for references

# Or using latexmk (recommended)
latexmk -pdf Report.tex
```

## Current Analysis Summary (Part 1)

### Best Systems by Metric:
- **P@10**: System 3 (0.410, tied with 5 & 6)
- **R@50**: System 2 (0.867)
- **R-Precision**: System 3 (0.449, tied with 6)
- **AP**: System 3 (0.451)
- **nDCG@10**: System 3 (0.420)
- **nDCG@20**: System 3 (0.511)

### Statistical Significance:
**All p-values > 0.05** - No significant differences detected

This is due to:
1. Limited number of queries (n=10)
2. High variance in system performance across queries
3. Multiple tied scores
4. Similar retrieval strategies among top systems

### Practical Recommendation:
Despite lack of statistical significance, **System 3** is the most reliable choice as it consistently achieves highest/near-highest scores across multiple metrics.

## Page Limit Compliance

The template is designed to stay within 6 pages:
- Section 1: ~1 page ✓
- Section 2: ~1 page ✓
- Section 3: ~1.5 pages (when filled)
- Section 4: ~1.5 pages (when filled)
- Conclusion: ~0.5 page

**Total: ~5.5 pages** (leaves room for adjustments)

## Tips for Writing

1. **Be concise**: The 6-page limit is strict
2. **Use tables effectively**: They present data compactly
3. **Focus on insights**: Don't just report numbers, explain what they mean
4. **Connect sections**: Show how error analysis led to improvements
5. **Be honest**: It's okay if some improvements didn't work - discuss why!

## Required LaTeX Packages

All necessary packages are already included:
- `booktabs` - Professional-looking tables
- `amsmath` - Mathematical notation
- `hyperref` - Clickable references
- `array`, `multirow` - Advanced table formatting
- `caption`, `subcaption` - Figure/table captions

