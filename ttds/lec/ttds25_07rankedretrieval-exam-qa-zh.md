# TTDS 07 Ranked Retrieval 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 ranked retrieval、Jaccard、TF-IDF、DF/CF/IDF、Vector Space Model、cosine similarity、length normalization、SMART notation 和 ranking output。

## 高频核心题

### Q1. Ranked retrieval 和 Boolean retrieval 的核心区别是什么？

A：Boolean retrieval 只给 match/no match；ranked retrieval 给每个 document 一个 score，并按分数排序返回 top-k。

### Q2. 为什么多数用户不适合 Boolean retrieval？

A：多数用户不会写复杂 Boolean queries，也不愿查看大量无排序结果。

### Q3. Ranked retrieval 为什么适合 web search？

A：它可处理 free text queries，并只展示最可能满足需求的 top results。

### Q4. Ranked retrieval 的核心输出是什么？

A：对 query 的 ranked list，例如 `(query_id, document_id, score)`。

### Q5. Ranked retrieval 的 score 表示什么？

A：document 与 query 的匹配程度或相关性估计。

## Jaccard Coefficient

### Q6. Jaccard coefficient 是什么？

A：衡量两个集合 overlap 的指标，公式为 `|A ∩ B| / |A ∪ B|`。

### Q7. Jaccard(A,A) 等于多少？

A：1。

### Q8. 如果 `A ∩ B = ∅`，Jaccard 等于多少？

A：0。

### Q9. Jaccard 在 IR 中的局限是什么？

A：不考虑 term frequency，所有 terms 权重相同，也缺少好的 length normalization。

### Q10. 为什么 Jaccard 会低估 rare terms 的重要性？

A：它把 “the” 和更有信息量的 rare term 同等对待。

## TF、DF、CF、IDF

### Q11. TF 是什么？

A：`tf(t,d)` 是 term `t` 在 document `d` 中出现次数。

### Q12. TF 的直觉是什么？

A：term 在 document 中出现越多，该 term 对该 document 越重要。

### Q13. DF 是什么？

A：`df(t)` 是包含 term `t` 的 documents 数量。

### Q14. CF 是什么？

A：`cf(t)` 是 term `t` 在整个 collection 中出现的总次数。

### Q15. DF 和 CF 的区别是什么？

A：DF 统计出现在哪些 documents；CF 统计总 occurrences。`df(t) <= N`，而 `cf(t)` 可大于 `N`。

### Q16. IDF 是什么？

A：inverse document frequency，用于衡量 term 的 informativeness。

### Q17. IDF 的公式是什么？

A：`idf(t) = log10(N / df(t))`。

### Q18. 为什么 IDF 使用 log scale？

A：dampen IDF effect，避免 rare terms 权重过度极端。

### Q19. 高频词的 IDF 为什么低？

A：它们出现在很多 documents 中，区分能力低。

### Q20. 如果 term 出现在所有 N 个 documents 中，IDF 是多少？

A：`log10(N/N)=0`。

## TF-IDF

### Q21. TF-IDF 的核心思想是什么？

A：term weight 随文档内出现次数增加，随 collection 中稀有程度增加。

### Q22. 讲义中的 TF-IDF term weight 是什么？

A：`w_{t,d} = (1 + log10 tf(t,d)) * log10(N / df(t))`。

### Q23. 为什么 TF 使用 `1 + log(tf)`？

A：第一出现很重要，后续出现仍增加权重但边际收益递减。

### Q24. Query-document score 如何由 TF-IDF 得到？

A：对 query 和 document 共有 terms 的 weights 求和，或在 VSM 中计算 cosine similarity。

### Q25. TF-IDF 为什么是强 baseline？

A：简单、可解释、有效，并且易于大规模实现。

## Vector Space Model

### Q26. Vector Space Model 是什么？

A：把 documents 和 queries 表示为同一 term space 中的向量，用向量相似度衡量匹配。

### Q27. 为什么不用 Euclidean distance 直接比较 document vectors？

A：文档长度不同会导致距离受长度影响；把 document 复制一遍语义不变但 Euclidean distance 会变大。

### Q28. VSM 中为什么比较 angle/cosine？

A：cosine 关注方向相似度，能减少长度差异影响。

### Q29. Cosine similarity 越大表示什么？

A：query vector 和 document vector 方向越接近，相关性越高。

### Q30. L2 normalization 是什么？

A：把向量每个分量除以向量 L2 norm，使其成为 unit vector。

### Q31. L2 normalization 的效果是什么？

A：使长短 documents 更可比；文档和其重复拼接版本 normalization 后方向相同。

### Q32. Cosine similarity 对 normalized vectors 如何计算？

A：直接做 dot product：`cos(q,d)=sum_i q_i d_i`。

### Q33. 非 normalized vectors 的 cosine 公式是什么？

A：`cos(q,d)= (q · d) / (||q|| ||d||)`。

### Q34. VSM ranking 的步骤是什么？

A：把 query 和 documents 表示为 weighted TF-IDF vectors，计算 cosine similarity，按 score 排序返回 top-k。

## SMART 与实践

### Q35. SMART notation 是什么？

A：描述 query 和 document 使用的 term weighting scheme 的记号，格式如 `ddd.qqq`。

### Q36. 标准 weighting scheme `lnc.ltc` 表示什么？

A：一种常用 SMART weighting 组合，documents 和 queries 使用不同 tf/idf/normalization 设置。

### Q37. Ranked retrieval output 通常包含什么字段？

A：query id、document id 和 score。

### Q38. Lab/CW 中的 OR-style scoring 直觉是什么？

A：只要 document 含有 query 中某些 terms 就可得分，score 是共有 terms 权重的加和。

### Q39. 为什么 top-k 返回能解决 large result sets？

A：用户通常只看前几个结果，系统无需展示所有 match。

## 考前作答模板

### Q40. 如果考“TF-IDF”，应该怎么答？

A：定义 TF、DF、IDF，写 `w=(1+log tf)*log(N/df)`，说明频繁出现在文档中且在集合中稀有的词权重大。

### Q41. 如果考“VSM + cosine”，应该怎么答？

A：说明 query/document 是 term-weight vectors，用 cosine 比较方向相似度，通过 L2 normalization 降低文档长度影响。

### Q42. 如果考“Jaccard vs TF-IDF”，应该怎么答？

A：Jaccard 只看 set overlap，不考虑 frequency 和 term informativeness；TF-IDF 同时考虑文档内频次和集合稀有度。

