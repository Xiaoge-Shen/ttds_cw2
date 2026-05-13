# TTDS 05 Indexing 1 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 Boolean search、document vectors、term-document matrix、inverted index、postings lists、index construction、query processing、phrase search 和 positional/proximity index。

## 高频核心题

### Q1. Index 的核心作用是什么？

A：让系统能快速找到 term 出现的位置，而不是每次 query 都 sequential scan 整个 collection。

### Q2. 为什么 grep/find 不适合大规模 search engine？

A：它们通常线性扫描文本，对大规模集合和高查询量不可行。

### Q3. Book index 和 IR index 的共同点是什么？

A：都用 lookup structure 快速找到词项对应的位置或文档。

### Q4. IR index 和 book index 的区别是什么？

A：IR index 面向大规模机器检索，可存 docID、frequency、positions 等信息，并支持 query processing。

### Q5. Preprocessing output 如何进入 index？

A：原始文本经过 tokenisation、normalisation、stopping、stemming 等处理后，processed terms 被加入 index。

## Document Vector 与 Collection Matrix

### Q6. Document vector 是什么？

A：把 document 表示为向量，维度是 terms，值可为 binary 或 term frequency。

### Q7. Collection matrix 是什么？

A：所有 document vectors 组成的 term-document 或 document-term 矩阵。

### Q8. Term-document incidence matrix 是什么？

A：用 0/1 表示每个 term 是否出现在每个 document 中的矩阵。

### Q9. Term frequency matrix 和 incidence matrix 的区别是什么？

A：incidence matrix 只记录是否出现；frequency matrix 记录出现次数。

### Q10. 为什么 collection matrix 通常 extremely sparse？

A：vocabulary 很大，但每个 document 只包含少量 terms，大多数 term-document entries 为 0。

### Q11. Heaps' law 和 Zipf's law 如何解释 matrix sparsity？

A：Heaps 说明 unique terms 很多，Zipf 说明大量 terms 只出现一次，因此矩阵绝大多数位置为 0。

## Boolean Search

### Q12. Boolean retrieval 是什么？

A：document 对 query 要么 match 要么不 match，通常用 AND、OR、NOT 组合 terms。

### Q13. Boolean query 的例子是什么？

A：`Brutus AND Caesar AND NOT Calpurnia`。

### Q14. Boolean search 如何用 incidence matrix 计算？

A：对对应 term rows 做逻辑运算，例如 AND 取交集，NOT 取补集。

### Q15. Boolean retrieval 的优点是什么？

A：精确、可解释，适合专家用户和严格约束场景。

### Q16. Boolean retrieval 的缺点是什么？

A：不会 ranking，query 写法困难，结果可能过多或过少。

## Inverted Index

### Q17. Inverted index 是什么？

A：从 term 映射到包含该 term 的 document list 的数据结构。

### Q18. Inverted index 为什么是 collection matrix 的 sparse representation？

A：它只存储出现过的 term-document pairs，而不存大量 0。

### Q19. Posting 是什么？

A：postings list 中的一条记录，通常包含 docID，可包含 term frequency 或 positions。

### Q20. Postings list 是什么？

A：某个 term 对应的全部 postings，通常按 docID 排序。

### Q21. Dictionary 在 inverted index 中是什么？

A：term vocabulary 及其 metadata，例如 document frequency 和指向 postings list 的 pointer。

### Q22. DocID 为什么通常排序？

A：排序后可用 linear merge 高效求交集/并集。

### Q23. Boolean inverted index 存什么？

A：每个 term 存包含它的 docIDs list。

### Q24. Frequency inverted index 存什么？

A：每个 term 存 `(DocID, count(term))` tuples。

### Q25. 为什么 frequency 信息重要？

A：ranked retrieval 中 term frequency 是 scoring 的重要特征。

## Index Construction

### Q26. Inverted index construction 的步骤是什么？

A：tokenize documents，normalise tokens，产生 `(term, docID)` pairs，按 term/docID 排序，合并重复项，生成 dictionary 和 postings。

### Q27. Step 1 term sequence 做什么？

A：从文档生成 `(term, docID)` 序列。

### Q28. Step 2 sorting 做什么？

A：按 term 然后 docID 排序，使相同 term 和相同 docID 的 entries 聚集。

### Q29. Step 3 posting 做什么？

A：合并同一 document 内重复 term，分离 dictionary/postings，并加入 df 信息。

### Q30. Document frequency `df` 是什么？

A：包含某 term 的 documents 数量。

### Q31. Collection frequency `cf` 和 df 有何不同？

A：cf 是 term 在 collection 中出现总次数；df 是包含该 term 的文档数。

## Query Processing

### Q32. Query `{ink AND wink}` 如何处理？

A：加载 ink 和 wink 的 postings lists，然后做 intersection/linear merge。

### Q33. Linear merge 的复杂度是什么？

A：`O(n)`，其中 `n` 是 query terms postings 总长度。

### Q34. AND 操作对应什么集合操作？

A：postings lists 的 intersection。

### Q35. OR 操作对应什么集合操作？

A：postings lists 的 union。

### Q36. NOT 操作对应什么集合操作？

A：posting list 的 complement，通常与 AND 一起使用。

### Q37. 为什么 query processing 中先处理短 postings list 有帮助？

A：短 list 交集更快，并可快速减少候选文档数。

## Phrase 与 Proximity

### Q38. Phrase search 比 Boolean AND 多了什么要求？

A：terms 不仅要都出现，还必须按顺序相邻出现。

### Q39. Bi-gram index 是什么？

A：把相邻词对作为 index term，例如 `pink_ink`。

### Q40. Bi-gram index 的优点是什么？

A：phrase query 查找快。

### Q41. Bi-gram index 的问题是什么？

A：index size 会膨胀，且无法自然支持任意长度 phrase 或 proximity query。

### Q42. Positional/proximity index 是什么？

A：在 postings 中存储 term 在 document 中的位置，从而支持 phrase 和 proximity search。

### Q43. Positional posting 通常长什么样？

A：例如 `(DocID, pos1, pos2, ...)` 或 `(DocID, term position)`。

### Q44. Phrase query `"pink ink"` 如何用 positional index 判断？

A：先找同时包含 pink 和 ink 的 docs，再检查 `pos(ink) - pos(pink) == 1`。

### Q45. Proximity query 如何判断？

A：先做 AND 找候选文档，再检查两个 term positions 的距离是否满足 `|pos2 - pos1| <= n`。

### Q46. Proximity index 的代价是什么？

A：存储 positions 增加 index size，但支持更强的 phrase/proximity retrieval。

## 考前作答模板

### Q47. 如果考“inverted index”，应该怎么答？

A：定义为 term -> postings list 的稀疏索引结构，posting 通常含 docID、frequency、positions，用于快速检索含 term 的文档。

### Q48. 如果考“index construction”，应该怎么答？

A：tokenize/normalise 后生成 `(term, docID)`，排序，合并重复，生成 dictionary、df 和 postings lists。

### Q49. 如果考“phrase search vs proximity search”，应该怎么答？

A：phrase search 要求相邻且顺序固定；proximity search 只要求 terms 在窗口距离内，通常依赖 positional index。

