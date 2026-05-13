# TTDS 10 IR Evaluation 2 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 test collection 创建、TREC、topic vs query、relevance judgments、pooling、Cohen's kappa、web search evaluation、A/B testing 和 significance testing。

## 高频核心题

### Q1. 本讲关注 IR evaluation 的什么问题？

A：如何创建 test collection、topic 与 query 的区别、relevance judgments、pooling 和统计显著性。

### Q2. Reusable IR test collection 包含什么？

A：document collection、topics/information needs、relevance judgments 和 evaluation measure。

### Q3. Web search 公司如何评估自己的系统？

A：使用 traffic、click/session logs、人工标注 selected queries、A/B testing 等。

### Q4. 为什么非 web search 往往需要 evaluation campaigns？

A：构建 test collection 成本高，evaluation campaigns 能集中创建 collection、topics 和 judgments。

### Q5. IR evaluation campaigns 的目的是什么？

A：为科研社区提供标准任务和 test collections，以比较和推动 IR 方法发展。

## TREC 与 Topics

### Q6. TREC 是什么？

A：Text REtrieval Conference，由 NIST 赞助，自 1992 年起组织的主要 IR evaluation campaign。

### Q7. TREC track 是什么？

A：围绕特定文档类型、领域或任务的检索评测任务集合，如 Medical、Legal、Microblog、CLIR。

### Q8. 其他 evaluation campaigns 有哪些？

A：CLEF、NTCIR、FIRE 等。

### Q9. TREC collection 通常包含什么？

A：大量 documents，常以 `<DOC>`, `<DOCNO>`, `<TEXT>` 等格式组织。

### Q10. Topic 和 query 有什么区别？

A：query 是简短查询文本；topic 还包含 description 和 narrative，明确信息需求和 relevance criteria。

### Q11. TREC topic 通常有哪些字段？

A：num、title/query、description 和 narrative。

### Q12. Narrative 的作用是什么？

A：说明什么内容应被视为 relevant 或 non-relevant，指导人工 judgment。

### Q13. 为什么 topic 比 query 更适合评测？

A：它减少信息需求歧义，使 assessors 能更一致判断 relevance。

## Relevance Judgments 与 Pooling

### Q14. 为什么 exhaustive relevance assessment 不现实？

A：topics 和 documents 数量太大，例如 50 topics × 1M+ documents。

### Q15. 为什么 random sampling 不适合找 relevant documents？

A：相关文档通常很稀少，随机采样可能一个 relevant doc 都抽不到。

### Q16. Pooling 的核心思想是什么？

A：让多个 IR systems 提交 top results，把这些候选合并成 pool，再人工判断。

### Q17. Pooling 的典型步骤是什么？

A：系统提交 top 1000，每个系统 top 100 入 pool，去重并随机排序，由 topic creator 判断，未评估文档视为 irrelevant。

### Q18. Pooling 为什么需要多个 reasonable systems？

A：不同系统会找出不同 relevant docs，合并后能覆盖更多相关文档。

### Q19. Pooling 为什么要求 systems 不都一样？

A：如果系统高度相似，pool 覆盖面小，会漏掉其他 relevant documents。

### Q20. 未评估 documents 通常如何处理？

A：在 pooling evaluation 中通常视为 irrelevant。

### Q21. Pooling 是否必须 exhaustive 才可靠？

A：不一定。研究表明系统相对排名通常仍较稳定。

### Q22. 未参与 pooling 的系统能否被评估？

A：可以，经典研究表明大规模 IR 实验结果通常仍可靠。

## Inter-annotator Agreement

### Q23. 为什么 relevance judgments 可能需要多个 assessors？

A：relevance 有主观性，同一文档不同人可能判断不同。

### Q24. Cohen's kappa 用来衡量什么？

A：衡量两个 judges 的 agreement，并扣除 chance agreement。

### Q25. Cohen's kappa 的公式是什么？

A：`κ = (P(A) - P(E)) / (1 - P(E))`。

### Q26. `P(A)` 是什么？

A：judges 实际一致的比例。

### Q27. `P(E)` 是什么？

A：按 chance 预期的一致比例。

### Q28. `κ=1` 表示什么？

A：total agreement。

### Q29. `κ=0` 表示什么？

A：只有 chance agreement。

### Q30. `κ<0` 表示什么？

A：比随机一致性更差。

### Q31. Kappa 大于 0.8 通常表示什么？

A：good agreement。

### Q32. Kappa 小于 0.67 表示什么？

A：judgments 作为 evaluation basis 可能可疑。

## Web Search Evaluation

### Q33. 为什么 web search 中 recall 难测？

A：web collection 巨大且不断变化，几乎不可能知道所有 relevant pages。

### Q34. Web search 常用什么 effectiveness measures？

A：precision@top k、nDCG、clickthrough、lab user studies 和 A/B testing。

### Q35. Clickthrough 单次是否可靠？

A：单次 click 不可靠，但 aggregate click data 可提供有用信号。

### Q36. A/B testing 的目的是什么？

A：测试单个 innovation 是否提升真实用户满意度。

### Q37. A/B testing 如何做？

A：大多数用户使用 old system，小比例流量进入 new system，用自动指标如 first-result clickthrough 比较。

### Q38. 为什么大型搜索引擎重视 A/B testing？

A：它直接在真实用户环境中测量变化对用户行为和满意度的影响。

## Significance Testing

### Q39. 为什么平均分更高还不足以说明系统更好？

A：提升可能来自随机波动，需要 significance test 判断差异是否可靠。

### Q40. Null hypothesis 是什么？

A：两个观察现象无关系或两个系统无真实差异。

### Q41. Significance test 的目标是什么？

A：在足够证据下拒绝 null hypothesis，支持系统 B 比 A 更有效。

### Q42. Test power 是什么？

A：当真实存在差异时，测试正确拒绝 null hypothesis 的概率。

### Q43. 增加 query 数量对 test power 有什么影响？

A：通常会提高 test power。

### Q44. Significance test 的基本步骤是什么？

A：计算每个 query 的 effectiveness，比较系统差异，计算 test statistic 和 p-value，若 `p <= α` 则拒绝 null。

### Q45. p-value 表示什么？

A：在 null hypothesis 为真时观察到当前或更极端差异的概率。

### Q46. 常用 significance level `α` 是多少？

A：通常为 0.05 或更小。

### Q47. t-test 的假设是什么？

A：每个 query 上 effectiveness difference 的样本来自 normal distribution。

### Q48. Wilcoxon test 什么时候有用？

A：当不想假设 differences 服从 normal distribution 时。

### Q49. 为什么用 AP 而不是 MAP 做 per-query significance test？

A：MAP 已经是 query 平均值；significance test 需要每个 query 的 effectiveness values。

### Q50. 如果 B 平均分高但 p=0.306，说明什么？

A：不能认为 B 显著优于 A，二者 statistically indistinguishable。

## 考前作答模板

### Q51. 如果考“pooling”，应该怎么答？

A：说明多个系统提交 top results，合并去重形成 pool，由 assessors 判断；未评估文档视为 irrelevant，适合避免 exhaustive assessment。

### Q52. 如果考“topic vs query”，应该怎么答？

A：query 是短文本；topic = query/title + description + narrative，用来明确信息需求和 relevance criteria。

### Q53. 如果考“significance test”，应该怎么答？

A：说明平均提升不足以证明改进；需 per-query scores、test statistic、p-value 和 `α=0.05`，常用 two-tailed t-test 或 Wilcoxon。

