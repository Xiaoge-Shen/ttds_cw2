# TTDS 09 IR Evaluation 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 IR 作为实验科学、Cranfield paradigm、test collection、precision/recall/F-measure、P@k、R-precision、AP/MAP、graded relevance、DCG/nDCG。

## 高频核心题

### Q1. 为什么说 IR 是 experimental science？

A：IR 需要提出 hypothesis、设计实验、与 baseline 比较、检验显著性、报告结果并迭代。

### Q2. IR evaluation 的目的是什么？

A：判断系统是否真的能更好满足用户 information need，并比较不同 retrieval methods。

### Q3. 为什么实验要有 baseline？

A：baseline 是 control，用来判断新方法是否真正带来提升。

### Q4. Evaluation 驱动什么？

A：它帮助识别哪些技术有效、哪些无效，从而决定研究和工程优化方向。

### Q5. IR evaluation 的三维度是什么？

A：Effectiveness、efficiency 和 usability。

### Q6. Effectiveness 衡量什么？

A：返回文档的相关性和质量。

### Q7. Efficiency 衡量什么？

A：retrieval time、indexing time、index size 等系统效率。

### Q8. Usability 衡量什么？

A：用户学习成本、灵活性和 novice/expert users 的使用体验。

## Cranfield Paradigm 与 Test Collection

### Q9. Cranfield paradigm 是什么？

A：在固定 document collection、queries/topics 和 relevance judgments 上，对 IR system 输出做 automatic effectiveness evaluation。

### Q10. Reusable IR test collection 包含什么？

A：document collection、sample information needs/topics、known relevance judgments 和 evaluation measure。

### Q11. Document collection 应满足什么要求？

A：应代表目标 IR task，考虑 size、sources、genre、topics 等。

### Q12. Information need sample 应满足什么要求？

A：应 randomized 和 representative，通常 formalized as topic statements。

### Q13. Relevance judgments 是什么？

A：人工判断每个 topic-document pair 是否相关，可为 binary 或 graded。

### Q14. Good effectiveness measure 应满足什么？

A：捕捉用户需求、可复现、可比较、最好为 single number，并对其他 collection/query 有预测价值。

## Set-based Measures

### Q15. Set-based measures 适合什么 retrieval？

A：适合 Boolean search 或不考虑 ranking 的 retrieved result sets。

### Q16. Precision 是什么？

A：retrieved documents 中 relevant 的比例，`P = TP / (TP + FP)`。

### Q17. Recall 是什么？

A：所有 relevant documents 中被 retrieved 的比例，`R = TP / (TP + FN)`。

### Q18. Precision 的直觉是什么？

A：系统返回的结果有多“干净”，即 top/retrieved docs 中有多少相关。

### Q19. Recall 的直觉是什么？

A：系统找回了多少所有相关文档。

### Q20. Precision 和 recall 的 tradeoff 是什么？

A：retrieve 更多文档通常提高 recall，但可能降低 precision。

### Q21. Accuracy 为什么不适合 IR？

A：相关文档极少，TN 巨大，系统即使几乎不找相关文档也可能 accuracy 很高。

### Q22. F1 是什么？

A：precision 和 recall 的 harmonic mean，`F1 = 2PR/(P+R)`。

### Q23. F-beta 是什么？

A：`Fβ = ((β^2+1)PR)/(β^2P+R)`，通过 `β` 控制 P/R 权重。

### Q24. `β=1` 表示什么？

A：precision 和 recall 同等重要。

### Q25. `β=5` 表示什么？

A：recall 比 precision 更重要，讲义说 recall five times more important。

## Ranked Retrieval Measures

### Q26. 为什么 ranked IR 不能只用 P/R/F？

A：两个系统可能 P/R/F 相同，但 relevant documents 的排名位置不同，用户体验不同。

### Q27. Precision@K 是什么？

A：只看 ranked list 前 K 个结果，计算其中 relevant 的比例。

### Q28. P@K 适合什么场景？

A：用户通常只看 top K 的场景，如 web search。

### Q29. P@K 的局限是什么？

A：不同 queries 的 relevant docs 数量不同，直接平均可能不好；也不考虑 K 后相关文档。

### Q30. R-Precision 是什么？

A：若某 query 有 `r` 个 relevant docs，则 R-precision = P@r。

### Q31. R-Precision 的直觉是什么？

A：考察理想情况下前 `r` 个位置是否能装下所有 relevant docs。

### Q32. 用户什么时候停止查看 ranked list 的假设是什么？

A：用户会一直看，直到 information need 被满足，通常假设会在某个 relevant document 处停止。

## AP 与 MAP

### Q33. Average Precision (AP) 是什么？

A：在每个 relevant document 出现的位置计算 precision@k，然后对这些 precision 值取平均。

### Q34. AP 的公式是什么？

A：`AP = (1/r) * sum_{k=1}^n P(k) * rel(k)`。

### Q35. AP 公式中 `r` 是什么？

A：该 query 的 relevant documents 总数。

### Q36. AP 公式中 `rel(k)` 是什么？

A：rank k 的文档是否 relevant，相关为 1，不相关为 0。

### Q37. MAP 是什么？

A：Mean Average Precision，即所有 queries 的 AP 平均值。

### Q38. MAP 的公式是什么？

A：`MAP = (1/Q) * sum_q AP(q)`。

### Q39. MAP 为什么常用于 IR？

A：它同时奖励 early relevant documents 和找回多个 relevant documents，是多数 IR task 常用 measure。

### Q40. 当 `r=1` 时，MAP 和什么等价？

A：MRR 的思想等价，因为只看第一个 relevant document 的 reciprocal rank。

### Q41. MAP 的 relevance 类型是什么？

A：binary relevance，即 relevant/irrelevant。

## Graded Relevance 与 nDCG

### Q42. 为什么需要 graded relevance？

A：有些 relevant documents 比其他 relevant documents 更有用，binary relevance 无法区分。

### Q43. Graded relevance 的例子是什么？

A：Perfect 4、Excellent 3、Good 2、Fair 1、Bad 0。

### Q44. DCG 的两个假设是什么？

A：高相关文档更有用；rank 越低，文档越不可能被用户看到，所以 gain 要 discount。

### Q45. DCG 是什么？

A：Discounted Cumulative Gain，累积 graded relevance gain，并按 rank 进行 discount。

### Q46. DCG 为什么用 discount？

A：用户更关注高排名结果，低排名结果贡献应减少。

### Q47. DCG@k 的基本形式是什么？

A：从 rank 1 到 k 累积 gain，后续 ranks 通常按 `1/log2(rank)` discount。

### Q48. nDCG 是什么？

A：Normalized DCG，用实际 DCG 除以 ideal ranking 的 iDCG。

### Q49. nDCG@k 的公式是什么？

A：`nDCG@k = DCG@k / iDCG@k`。

### Q50. nDCG 的范围是什么？

A：通常不超过 1，ideal ranking 的 nDCG 为 1。

### Q51. 为什么 nDCG 常用于 web search？

A：它能处理 graded relevance，并强烈关注 top ranks。

## 考前作答模板

### Q52. 如果考“precision vs recall”，应该怎么答？

A：precision 是 retrieved 中有多少 relevant；recall 是 relevant 中找回多少；两者常 trade off，accuracy 因 TN 巨大不适合 IR。

### Q53. 如果考“MAP”，应该怎么答？

A：AP 在每个 relevant rank 计算 P@k 并平均；MAP 是 across queries 的 AP 平均，关注 early relevant documents，使用 binary relevance。

### Q54. 如果考“nDCG”，应该怎么答？

A：DCG 用 graded relevance 并按 rank discount；nDCG 用 ideal DCG 归一化，使不同 queries 可比较。

