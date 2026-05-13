# TTDS 13 Comparing Text Corpora 2 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 plate notation、unigram model、mixture of unigrams、pLSI、LDA、model inference、Gibbs sampling，以及 annotation + classification 的 corpus analysis workflow。

## 高频核心题

### Q1. 本讲的核心主题是什么？

A：深入 topic modelling，尤其是 LDA 的概率模型、inference 和 Gibbs sampling。

### Q2. Plate notation 用来表示什么？

A：用图形方式表示 probabilistic graphical model 中的变量、依赖关系和重复结构。

### Q3. Plate 中的 N/M 通常表示什么？

A：某个随机过程重复 N 次或 M 次，例如 document 中 N 个 words、corpus 中 M 个 documents。

### Q4. 为什么要从 unigram model 讲起？

A：它是最简单的文本生成模型，有助于逐步理解更复杂的 topic models。

## Unigram Model

### Q5. Unigram model 是什么？

A：每个 word 独立地从同一个 vocabulary probability distribution 中生成。

### Q6. Unigram model 中 `w` 表示什么？

A：一个 word；document 可看作 words 的 vector/sequence。

### Q7. Unigram model 如何计算句子概率？

A：把每个 word 的概率相乘。

### Q8. “My dog barked at another dog.” 的 unigram 概率如何算？

A：按每个词概率相乘，例如 `.1 * .05 * .03 * .1 * .04 * .05`。

### Q9. 为什么更高 text probability 不一定表示更好模型？

A：目标不是给所有文本高概率，而是给真实 documents 高概率、给 noise 低概率。

### Q10. Unigram model 的主要局限是什么？

A：它不区分 documents/topics，所有 words 都来自同一个 distribution。

## Mixture of Unigrams

### Q11. Mixture of unigrams model 是什么？

A：每个 document 先选择一个 topic `z`，再从该 topic 的 word distribution 生成所有 words。

### Q12. Mixture of unigrams 中 `z` 表示什么？

A：document 的 topic。

### Q13. Mixture of unigrams 如何计算 document probability？

A：对每个可能 topic，计算 `P(z) * product_i P(w_i|z)`，再把 topics 的概率相加。

### Q14. Mixture of unigrams 相比 unigram 的改进是什么？

A：它允许不同 documents 来自不同 topics。

### Q15. Mixture of unigrams 的局限是什么？

A：每个 document 只能有一个 topic，不适合一篇文档包含多个主题的情况。

## pLSI

### Q16. pLSI 是什么？

A：Probabilistic Latent Semantic Indexing，通过 document-specific topic distributions 生成 words。

### Q17. pLSI 中 `d` 表示什么？

A：document ID，模型直接依赖 document identity。

### Q18. pLSI 如何比 mixture of unigrams 更灵活？

A：一个 document 可混合多个 topics，而不是只有一个 topic。

### Q19. pLSI 的 joint probability 可如何理解？

A：先选 document，再按该 document 的 topic distribution 选 topic，再按 topic-word distribution 生成 word。

### Q20. pLSI 的局限是什么？

A：参数随 documents 增加，缺少对新 documents 的自然 generative process。

## LDA

### Q21. LDA 是什么？

A：Latent Dirichlet Allocation，一种生成式 topic model，每个 document 有 topic distribution，每个 topic 有 word distribution。

### Q22. LDA 中 `θ` 表示什么？

A：document-level distribution over topics。

### Q23. LDA 中 `α` 表示什么？

A：Dirichlet prior 的参数，控制 documents 中 topic distributions 的形状。

### Q24. LDA 中 `β` 表示什么？

A：topic-word distributions 的参数或 prior，控制 topics 中 words 的分布。

### Q25. LDA 中 `z` 表示什么？

A：每个 word token 的 latent topic assignment。

### Q26. LDA 中 `w` 表示什么？

A：observed word token。

### Q27. LDA 的生成过程是什么？

A：对每个 document 采样 topic distribution `θ`；对每个 word 采样 topic `z`，再从 topic 的 word distribution 采样 word `w`。

### Q28. LDA 相比 pLSI 的关键优势是什么？

A：用 Dirichlet prior 建模 documents 的 topic mixtures，更适合生成和处理 unseen documents。

### Q29. LDA 输出的两个核心矩阵是什么？

A：document-topic matrix `θ` 和 topic-word matrix `Φ`。

### Q30. `Φ` 表示什么？

A：topic-word probabilities，即每个 topic 生成各 words 的概率。

## Model Inference

### Q31. Model inference 的目标是什么？

A：从 corpus 学习 latent topic assignments 和模型参数，如 `Φ` 和 `θ`。

### Q32. 为什么 LDA exact inference intractable？

A：latent topic assignments 组合空间巨大，精确求后验计算不可行。

### Q33. 常用 approximate inference 方法有哪些？

A：Gibbs sampling 和 variational inference。

### Q34. Gibbs sampling 的目标是什么？

A：在给定 documents、`α`、`β` 的情况下，迭代采样每个 word 的 topic assignment，学习 `Φ` 和 `θ`。

### Q35. Gibbs sampling 的基本步骤是什么？

A：随机初始化 topics，计算 count matrices，反复对每个 word 取消当前 assignment、采样新 topic、更新 counts，直到收敛。

### Q36. Gibbs sampling 中为什么要 decrement count matrices？

A：在重新采样某个 word 的 topic 前，需要先移除它当前 assignment 对 counts 的影响。

### Q37. `α` 和 `β` 在 Gibbs sampling 中有什么作用？

A：作为 smoothing/prior，避免零概率并控制 topic/document 分布。

### Q38. Gibbs sampling 的结果是否完全确定？

A：不是。它是 probabilistic algorithm，结果依赖 random initialization 和 random samples。

### Q39. Gibbs sampling 何时停止？

A：重复直到 convergence 或达到预设迭代次数。

## Annotation + Classification

### Q40. Topic modelling 之外还可以如何做 corpus analysis？

A：annotation + classification。

### Q41. Traditional supervised learning workflow 是什么？

A：标注 representative samples，训练 classifier，然后应用到剩余数据。

### Q42. Transfer learning workflow 是什么？

A：找一个大而相似的数据集训练 classifier，可选地在小数据集 fine-tune，再应用到目标数据。

### Q43. Classification 后可以分析什么？

A：每个 class 的重要 features、常见 words/topics，以及 predicted classes 与其他 variables 的关系。

### Q44. Topic modelling 和 classification 的区别是什么？

A：topic modelling 通常 unsupervised，发现 latent topics；classification supervised，用预定义 labels 预测 classes。

### Q45. 什么时候用 topic modelling？

A：当没有预定义 labels、想探索 corpus themes 时。

### Q46. 什么时候用 classification？

A：当有明确 categories 和 labeled data，想给大量 documents 自动打标签时。

## 考前作答模板

### Q47. 如果考“LDA 生成过程”，应该怎么答？

A：每个 document 先从 Dirichlet prior 得到 topic distribution `θ`；每个 word 先采样 topic `z`，再从 topic-word distribution `Φ` 采样 observed word `w`。

### Q48. 如果考“unigram/mixture/pLSI/LDA 比较”，应该怎么答？

A：unigram 一个全局 word distribution；mixture 每文档一个 topic；pLSI 每文档 topic mixture 但依赖 docID；LDA 用 Dirichlet prior 为 documents 生成 topic mixtures。

### Q49. 如果考“Gibbs sampling”，应该怎么答？

A：随机初始化 topic assignments，迭代移除一个 token 的 assignment、根据当前 counts 采样新 topic、更新 counts，最后估计 `Φ` 和 `θ`。

