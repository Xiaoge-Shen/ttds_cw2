# Information Retrieval System Implementation Report

## 1. Introduction

This report presents the implementation of a comprehensive information retrieval system supporting both Boolean and Ranked retrieval methods. The system processes XML documents, builds an inverted index, and provides search capabilities using TF-IDF scoring.

## 2. Tokenization and Stemming Methods

### 2.1 Tokenization Process

The tokenization process follows a systematic approach:
1. **Text Normalization**: Convert all text to lowercase for case-insensitive matching
2. **Pattern Matching**: Use regular expression `[a-z]+` to extract alphabetic sequences
3. **Stopword Filtering**: Remove common English stopwords using a predefined list
4. **Token Validation**: Retain only valid tokens after pattern matching and stopword filtering

### 2.2 Stemming Implementation

The system employs the Porter Stemmer algorithm for morphological normalization:
- **Algorithm Choice**: Porter Stemmer selected for effectiveness in reducing words to root forms
- **Consistency**: Same stemming algorithm used for both indexing and query processing
- **Examples**: "corporation" → "corpor", "taxes" → "tax", "reduction" → "reduct"

## 3. Inverted Index Implementation

### 3.1 Index Structure

The inverted index uses a nested dictionary structure:
```
{term: {doc_id: [position1, position2, ...]}}
```

This provides efficient access to document frequency, term frequency, and position information.

### 3.2 Index Construction

1. **Document Processing**: Parse XML to extract headline and text content
2. **Token Processing**: Tokenize, stem, and filter documents
3. **Position Tracking**: Record each token's position within documents
4. **Index Building**: Map terms to documents and positions
5. **Persistence**: Save index to disk for reuse

## 4. Search Function Implementation

### 4.1 Boolean Search Engine

The Boolean search engine implements unified query processing:

**Query Parsing**: Identifies AND, OR, AND NOT, OR NOT operators with priority-based parsing
**Boolean Operations**: 
- AND: Set intersection for documents containing all terms
- OR: Set union for documents containing any terms
- AND NOT: Set difference
- OR NOT: Union with complement

**Phrase Processing**: Ensures consecutive term positions for phrase queries

### 4.2 Ranked Search Engine

The ranked search engine uses TF-IDF scoring:

**TF-IDF Calculation**:
- Term Frequency (TF): Normalized by document length
- Inverse Document Frequency (IDF): Logarithmic scaling based on document frequency
- Score: TF × IDF for each term, summed across query terms

**Query Processing**: Identifies candidate documents, computes scores, and sorts by relevance

### 4.3 Query Type Support

The system supports six query types:
1. Single Term, 2. OR Queries, 3. AND NOT Queries, 4. Phrase Queries, 5. Proximity Queries, 6. Complex Queries

## 5. System Architecture and Results

### 5.1 Design Decisions

- **Unified Processing**: Single entry point for all query types
- **Modular Design**: Separate engines for Boolean and Ranked retrieval
- **Performance Optimization**: Precomputed statistics and efficient data structures

### 5.2 Implementation Challenges

**Index Construction**: Building efficient index from large XML documents
**Query Complexity**: Supporting multiple query types with different requirements
**Performance**: Balancing accuracy with processing speed
**Phrase Matching**: Ensuring consecutive term positions

### 5.3 System Results

**Boolean Retrieval**: Successfully processes all query types (Query 1: 141 results, Query 5: 1,104 results)
**Ranked Retrieval**: Provides relevance scoring (Query 1: 1,459 documents, scores 0.2575-0.0012)
**Index Statistics**: 32,933 unique terms, 8.3MB index size, 5,000 documents processed

## 6. Challenge: Impact of Stopword Filtering

### 6.1 Experimental Setup

Modified system to process all terms without filtering common stopwords to evaluate their importance.

### 6.2 Observed Changes

**Retrieved Results**:
- Boolean: 2,020 → 1,997 results (-1.1%)
- Ranked: 14,272 → 34,946 results (+145%)
- Score reduction: Query 1 top score 0.2575 → 0.1556 (-40%)

**Processing Time**:
- Index Building: 30s → 17.74s (-41% faster, no filtering overhead)
- Ranked Queries: 2s → 9.15s (+358% slower due to more terms)

**Index Size**:
- File Size: 8.3MB → 14MB (+69% increase)
- Term Count: 32,933 → 33,220 (+0.9% increase)

### 6.3 Quality Impact

**Precision Degradation**: Ranked retrieval returned significantly more irrelevant documents
**Score Distribution**: Relevance scores became less discriminative
**Performance Impact**: Query processing became 4x slower

### 6.4 Conclusion

The challenge demonstrates the critical importance of stopword filtering. While removing stopwords increases recall, it severely degrades precision and performance, making it unsuitable for production systems.

## 7. System Improvements and Scaling

### 7.1 Performance Enhancements

1. **Index Compression**: Reduce storage requirements
2. **Caching**: Add query result caching for frequently accessed terms
3. **Parallel Processing**: Multi-threading for index construction and query processing

### 7.2 Algorithm Improvements

1. **BM25 Scoring**: Upgrade from TF-IDF to BM25 for better relevance ranking
2. **Query Expansion**: Implement synonym expansion and query reformulation
3. **Machine Learning**: Integrate learning-to-rank algorithms

### 7.3 Scalability Solutions

1. **Distributed Indexing**: For large datasets
2. **Sharding**: Partition index across multiple servers
3. **Real-time Updates**: Support incremental index updates

## 8. Conclusion

The implemented information retrieval system successfully demonstrates core IR concepts including tokenization, stemming, index construction, and multiple search paradigms. The system achieves good performance on Boolean queries and provides meaningful relevance scoring for ranked retrieval.

The challenge experiment clearly illustrates the importance of text preprocessing in information retrieval, showing that stopword filtering has profound effects on system performance, precision, and user experience.

Key learnings include the critical role of preprocessing in IR systems, the precision-recall tradeoff in ranked retrieval, and the importance of efficient data structures for large-scale text processing.
