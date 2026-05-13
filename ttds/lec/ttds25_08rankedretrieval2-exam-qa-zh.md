# TTDS 08 Ranked Retrieval 2 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 probabilistic models、Probability Ranking Principle、BM25、language model for IR、query likelihood、MLE、smoothing、Jelinek-Mercer smoothing 和 VSM/BM25/LM 的比较。

## 高频核心题

### Q1. 为什么说 VSM 是 heuristic？

A：VSM 没有显式 relevance probability，只用可调 weighting scheme 和 similarity measure，工程上有效但理论解释较弱。

### Q2. Probabilistic retrieval model 的目标是什么？

A：用概率论表示 IR 中的不确定性，估计 document 对 query 的 relevance probability。

### Q3. Probabilistic model 会显式定义什么？

A：random variables，如 relevance、query、document，以及它们的取值和 assumptions。

### Q4. Probability Ranking Principle 的核心是什么？

A：按 documents 对用户请求的 relevance probability 从高到低排序，在可用数据下能获得最优 effectiveness。

### Q5. PRP 中“概率估计越准越好”意味着什么？

A：`P_est(R|D)` 应尽可能接近真实但不可直接观测的 `P_true(R|D)`。

### Q6. 为什么 relevance probability 不等于直接看用户脑子？

A：系统无法观察真实意图，只能根据 query、document、session、context、user 等信号估计。

### Q7. `P_true(rel | Q,D)` 可如何理解？

A：在未观察用户/情境/任务中，document 会被判 relevant 的比例。

## BM25

### Q8. BM25 基于什么思想？

A：基于 probabilistic model 和 binary independence model 的扩展。

### Q9. Binary independence model 的假设是什么？

A：document 用 binary term features 表示，并假设 terms 条件独立，类似 Naive Bayes。

### Q10. BM25 为什么重要？

A：它在 TREC 中表现优异，是最常用、有效且强大的 ranking formula 之一。

### Q11. BM25 的主要组件有哪些？

A：tf component、idf component 和 document length normalization。

### Q12. BM25 中 `L_d` 表示什么？

A：document `d` 的 terms 数量，即 document length。

### Q13. BM25 中 `L_avg` 表示什么？

A：collection 中平均 document length。

### Q14. BM25 中 `k` 控制什么？

A：控制 term frequency saturation，讲义中 best practice 取 `k=1.5`。

### Q15. BM25 的 tf component 有什么特点？

A：term frequency 增加会提高权重，但增长会饱和，避免长文档/重复词过度加分。

### Q16. BM25 的 idf component 直觉是什么？

A：rare terms 权重大，出现在很多 documents 的 terms 权重小。

### Q17. BM25 如何处理 document length bias？

A：用 `L_d / L_avg` 做 length normalization，长文档不会因词多而过度占优。

### Q18. BM25 和 TF-IDF 的相似点是什么？

A：都奖励 document 内 query term 频繁出现，并奖励 collection 中稀有 terms。

### Q19. BM25 相比普通 TF-IDF 的优势是什么？

A：更明确处理 tf saturation 和 document length normalization，通常更稳健。

## Language Model for IR

### Q20. LM approach 的核心思想是什么？

A：如果 document language model 很可能生成 query，则该 document 是 query 的好匹配。

### Q21. Noisy-channel view 如何解释 IR？

A：用户有 information need，写出 query；系统根据 query 猜最可能对应的 relevant document。

### Q22. Query likelihood model 排序依据是什么？

A：按 `P(q | M_d)` 排序，其中 `M_d` 是 document `d` 的 language model。

### Q23. LM in IR 中为什么 `P(d|q)` 可转为 `P(q|d)`？

A：由 Bayes rule，`P(q)` 对所有 documents 相同，`P(d)` 常假设相同或作为 prior，因此排序可基于 `P(q|d)`。

### Q24. Document prior `P(d)` 可以包含什么信号？

A：document quality，如 PageRank。

### Q25. Language model 是什么？

A：对 vocabulary 上 strings 的 probability distribution。

### Q26. Unigram LM 的假设是什么？

A：query/document terms 独立生成，不考虑词序和上下文。

### Q27. Bigram/n-gram LM 与 unigram 的区别是什么？

A：n-gram 根据前 `n-1` 个词预测当前词，考虑局部上下文。

### Q28. Query likelihood 的条件独立公式是什么？

A：`P(q | M_d) = product_k P(t_k | M_d)`，也可按 term frequency 写成 `product_t P(t|M_d)^{tf(t,q)}`。

### Q29. MLE 如何估计 document LM 中 term probability？

A：`P(t|M_d) = tf(t,d) / |d|`。

### Q30. 为什么 MLE 会产生 zero probability 问题？

A：如果 query term 没出现在 document 中，则 `P(t|M_d)=0`，整个 query probability 变 0。

### Q31. 为什么 zero probability 不合理？

A：document text 只是 model 的 sample，未观察到的词不代表绝不可能出现。

## Smoothing

### Q32. Smoothing 的目标是什么？

A：给 unseen terms 分配非零概率，同时降低已见 terms 的概率估计。

### Q33. Smoothing 解决什么问题？

A：data sparsity 和 zero frequency 问题。

### Q34. Mixture model smoothing 的公式是什么？

A：`P(t|d) = λP(t|M_d) + (1-λ)P(t|M_c)`。

### Q35. `M_c` 是什么？

A：collection language model，也称 background LM。

### Q36. Unseen word 在 mixture model 中的概率是什么？

A：`(1-λ)P(t|M_c)`。

### Q37. Observed word 在 mixture model 中的概率是什么？

A：`λP(t|M_d) + (1-λ)P(t|M_c)`。

### Q38. Jelinek-Mercer smoothing 是什么？

A：用 parameter `λ` 线性混合 document LM 和 collection LM 的 smoothing 方法。

### Q39. 高 `λ` 会产生什么 retrieval 行为？

A：更 conjunctive-like，倾向检索包含所有 query words 的 documents。

### Q40. 低 `λ` 会产生什么 retrieval 行为？

A：更 disjunctive，更适合 long queries。

### Q41. 为什么正确设置 `λ` 很重要？

A：它控制 document evidence 和 collection background 的平衡，直接影响 ranking。

## 模型比较

### Q42. VSM 问的问题是什么？

A：query vector 和 document vector 对齐得多好。

### Q43. Probabilistic model 问的问题是什么？

A：document D 在 query Q 下的 relevance probability 是多少。

### Q44. LM model 问的问题是什么？

A：document language model 生成 query term sequence 的可能性有多大。

### Q45. BM25 和 query likelihood LM 的效果关系如何？

A：讲义指出 query likelihood LM 与 BM25 有相似 effectiveness，更 sophisticated methods 可超过 BM25。

### Q46. LM for IR 的三种可能方式是什么？

A：document LM 生成 query；query LM 生成 document；比较 query/document topic language models。

## 考前作答模板

### Q47. 如果考“PRP”，应该怎么答？

A：说明最佳排序应按 documents 对用户请求的 relevance probability 递减排序，前提是概率基于可用数据尽量准确估计。

### Q48. 如果考“BM25”，应该怎么答？

A：说明 BM25 是 probabilistic ranking formula，结合 saturated TF、IDF 和 document length normalization，`k≈1.5` 控制 tf saturation。

### Q49. 如果考“query likelihood LM”，应该怎么答？

A：为每个 document 建 unigram LM，计算 `P(q|M_d)`，用 MLE 估计 `P(t|M_d)=tf/|d|`，并用 smoothing 解决 unseen terms。

### Q50. 如果考“smoothing”，应该怎么答？

A：说明未见词不能给 0 概率；Jelinek-Mercer 用 `λP(t|M_d)+(1-λ)P(t|M_c)` 混合 document 和 collection probabilities。

