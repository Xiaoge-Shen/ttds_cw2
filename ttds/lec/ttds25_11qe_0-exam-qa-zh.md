# TTDS 11 Query Expansion 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 query expansion、thesaurus methods、relevance feedback、Rocchio algorithm、pseudo relevance feedback、query drift，以及 neural/dense retrieval 的关系。

## 高频核心题

### Q1. Query expansion 是什么？

A：自动向用户 query 添加相关 terms，以更好匹配 relevant documents。

### Q2. 为什么需要 query expansion？

A：用户 query 往往短、不完整或用词 suboptimal，而相关文档可能使用不同词表达同一需求。

### Q3. Stemming/lemmatisation 不能解决哪些匹配问题？

A：synonyms、abbreviations 和语义相关词，如 car/vehicle/automobile、US/USA/the states。

### Q4. Query expansion 的主要方法有哪些？

A：thesaurus、retrieved documents-based expansion、relevance feedback、pseudo relevance feedback 和 query logs。

### Q5. QE 的主要风险是什么？

A：加入错误 terms 会导致 query drift，降低 precision 或 recall。

## Thesaurus-based QE

### Q6. Thesaurus 是什么？

A：把词组织成 synonym sets 或 related word sets 的资源。

### Q7. Manual thesaurus 的例子是什么？

A：WordNet。

### Q8. Automatic thesaurus 可如何构建？

A：基于 word co-occurrence 或 parallel corpus translations。

### Q9. Co-occurrence thesaurus 的假设是什么？

A：在同一 document/paragraph 中共现的词可能语义相似或相关。

### Q10. Co-occurrence thesaurus 如何用 collection matrix 构建？

A：若 `A` 是 term-document matrix，可计算 `C = A A^T`，其中 `C_{u,v}` 表示 term u 和 v 的相似度。

### Q11. Co-occurrence thesaurus 的优点是什么？

A：unsupervised，不需要人工标注。

### Q12. Co-occurrence thesaurus 的缺点是什么？

A：更容易得到 related words 而不是真正 synonyms。

### Q13. Parallel corpus thesaurus 的核心思想是什么？

A：如果语言 X 中多个词翻译成语言 Y 中同一词，这些 X 词可能构成 synset。

### Q14. Parallel corpus method 是 supervised 还是 unsupervised？

A：需要 parallel corpus 作为训练数据，因此是 supervised/resource-dependent。

### Q15. Thesaurus-based QE 为什么常失败？

A：它通常忽略 context，同一词在不同语境下可能需要不同扩展。

### Q16. Thesaurus-based QE 什么时候更有效？

A：在专门领域，如 medical domain，有高质量 domain thesaurus 时更有效。

## Relevance Feedback

### Q17. Relevance feedback 是什么？

A：用户标记初始结果中哪些 relevant/non-relevant，系统据此改写 query。

### Q18. Relevance feedback 的基本流程是什么？

A：用户提交短 query，系统返回结果，用户标注相关性，系统用反馈计算更好的 information need representation，可迭代多轮。

### Q19. Relevance feedback 为什么符合用户认知？

A：用户可能难以写好 query，但通常更容易判断具体文档是否相关。

### Q20. Relevance feedback 可提升什么？

A：可提高 recall 和 precision，尤其在 recall 重要的场景下有用。

### Q21. Relevance feedback 的实际问题是什么？

A：用户不愿显式标注，长 query 也会增加系统成本和响应时间。

### Q22. 为什么 relevance feedback 后结果更难解释？

A：query 被加入很多新 terms，用户难以理解某文档为何被检索。

## Rocchio Algorithm

### Q23. Rocchio algorithm 的核心思想是什么？

A：把 query vector 向 relevant documents 的 centroid 移动，远离 non-relevant documents 的 centroid。

### Q24. Centroid 是什么？

A：一组 document vectors 的中心向量，`μ(C) = (1/|C|) sum_{d in C} d`。

### Q25. Rocchio 的理论 optimal query 是什么？

A：`q_opt = μ(C_rel) - μ(C_irrel)`。

### Q26. Rocchio 实践公式是什么？

A：`q_m = αq_0 + β(1/|D_rel|)sum_{d in D_rel}d - γ(1/|D_irrel|)sum_{d in D_irrel}d`。

### Q27. `q_0` 表示什么？

A：original query vector。

### Q28. `q_m` 表示什么？

A：modified query vector。

### Q29. `α` 控制什么？

A：保留 original query 的权重。

### Q30. `β` 控制什么？

A：positive feedback，即 relevant documents 对新 query 的影响。

### Q31. `γ` 控制什么？

A：negative feedback，即 non-relevant documents 对新 query 的排斥影响。

### Q32. 为什么实践中通常 `β > γ`？

A：positive feedback 更有价值，negative feedback 更容易引入不稳定或错误。

### Q33. 为什么有些系统设置 `γ=0`？

A：只使用 positive feedback，避免 negative feedback 导致负权重或错误排除。

### Q34. Rocchio 中为什么只选 top nt terms？

A：避免 query 过长，提高效率，并减少加入噪声 terms。

## Pseudo Relevance Feedback

### Q35. Pseudo relevance feedback 是什么？

A：系统假设初始检索结果 top k documents 是 relevant，无需用户标注，自动做 relevance feedback。

### Q36. PRF 的基本算法是什么？

A：先检索 ranked list，假设 top k relevant，用 Rocchio 等方法扩展 query，通常只用 positive feedback。

### Q37. PRF 解决了什么问题？

A：用户不愿提供 explicit feedback 的问题。

### Q38. PRF 的主要风险是什么？

A：如果 top k documents 不相关，扩展会把 query 带偏，产生 query drift。

### Q39. 多轮 PRF 为什么可能危险？

A：错误扩展会逐步累积，导致 query drift 越来越严重。

### Q40. PRF 适合哪些应用？

A：news search、social media search、web search 中 implicit feedback，以及许多无需语言资源的 IR 场景。

### Q41. PRF 对 patent search 为什么更难？

A：top documents 常不相关，patent text 本身晦涩，错误扩展风险高。

### Q42. PRF evaluation 应如何做？

A：测试不同 feedback docs 数 `nd` 和 terms 数 `nt`，与 no-PRF baseline 比较，并做 significance test。

### Q43. PRF 是否算 cheating？

A：不算。它是标准 unsupervised retrieval technique，但必须与 baseline 公平比较。

## Modern Term Representations

### Q44. Local representation 是什么？

A：term 是确定离散符号，Car、Vehicle、Hamster 都不同。

### Q45. Vector representation 的目标是什么？

A：让语义相近的 terms/sentences/images 向量更接近。

### Q46. Word embeddings 对 QE 有什么作用？

A：可用于 semantic similarity 和 query expansion，但静态 embeddings 缺乏上下文。

### Q47. BERT/Transformers 对 IR 有何影响？

A：contextualized embeddings 支持更深的 query-document understanding 和 reranking。

### Q48. Dense retrieval 的核心变化是什么？

A：用 dense embeddings 和 ANN search 进行端到端 neural retrieval。

### Q49. LLMs/RAG 如何影响 IR？

A：LLMs 可用于 query rewriting、reranking、document expansion，retrieval 成为 RAG 的核心组成。

### Q50. Term-based search 是否过时？

A：没有。Inverted index、BM25、PRF 仍高效、可扩展，常作为第一阶段 retrieval，再用 advanced models rerank。

## 考前作答模板

### Q51. 如果考“Rocchio”，应该怎么答？

A：写公式，解释 query 向 relevant centroid 移动、远离 non-relevant centroid；`α,β,γ` 控制原 query、正反馈、负反馈权重。

### Q52. 如果考“PRF”，应该怎么答？

A：说明假设 top k 初始结果 relevant，自动做 positive feedback；优点是不需用户标注，风险是 query drift。

### Q53. 如果考“thesaurus QE 为什么失败”，应该怎么答？

A：说明词表忽略上下文，容易扩展到 related 而非真正 relevant 的词，导致 precision/recall 不稳定。

