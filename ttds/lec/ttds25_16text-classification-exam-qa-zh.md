# TTDS 16 Text Classification 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 text classification 定义、classification types、rule-based 与 supervised classification、feature extraction/selection/weighting、word embeddings、pre-trained language models、zero/few-shot classification、evaluation、baselines、data split 和 hyperparameter optimisation。

## 高频核心题

### Q1. Text classification 是什么？

A：根据文本内容把 documents、articles、sentences、queries 等分类到预定义类别中。

### Q2. Classification 的一般定义是什么？

A：预测 data item 属于预定义有限 classes 中哪个 class 的任务，可表示为 `h: D -> C`。

### Q3. Classification 和 clustering 的区别是什么？

A：classification 的 classes 预先给定；clustering 的 groups 和数量事先未知。

### Q4. 什么任务不适合作为 classification？

A：类别成员可确定性计算的任务，例如判断自然数是否 prime。

### Q5. Text classification 的数据项可以是什么？

A：news articles、emails、sentences、queries、web pages 等完全或部分文本对象。

## Types of Classification

### Q6. Binary classification 是什么？

A：把 item 分到两个 classes 之一，例如 spam/not spam。

### Q7. Single-label multi-class classification 是什么？

A：从多个 classes 中选一个，例如 sports/politics/entertainment。

### Q8. Multi-label multi-class classification 是什么？

A：一个 item 可属于零个、一个或多个 classes，例如 ACM 分类体系。

### Q9. MLMC 常如何解决？

A：通常拆成多个 independent binary classification problems。

### Q10. Classification dimensions 有哪些？

A：topic、sentiment、language、genre、author、native language、gender、usefulness 等。

### Q11. Language identification 为什么是 text classification？

A：它把文本分到预定义语言类别中。

### Q12. Authorship attribution 为什么是 text classification？

A：它根据文本特征预测作者类别。

## Methods and Models

### Q13. Text classification 方法主要有哪些？

A：rule-based classification、supervised learning、word embeddings、pre-trained language models、zero/few-shot classification。

### Q14. Rule-based classification 是什么？

A：人工构建分类规则或 dictionaries，例如包含 Viagra/Cialis 则判为 spam。

### Q15. Rule-based classification 的优点是什么？

A：可解释、无需 labeled training data，适合明确规则场景。

### Q16. Rule-based classification 的缺点是什么？

A：构建和维护昂贵，coverage 差，recall 通常低。

### Q17. Supervised-learning classification 是什么？

A：使用人工标注样本训练 classifier，让模型学习新文本属于某 class 的特征。

### Q18. Supervised classification 的优点是什么？

A：标注样本通常比写复杂规则便宜，且更容易适应新类别和语义变化。

### Q19. Supervised classification pipeline 是什么？

A：label sample，extract features，train classifier，apply model to unlabelled documents。

## Feature Extraction

### Q20. 为什么文本要转换成 vectors？

A：大多数 learning algorithms 需要数值向量输入。

### Q21. Feature extraction 是什么？

A：确定哪些文本属性作为 vector dimensions。

### Q22. 最简单的 text feature 是什么？

A：Bag-of-words，每个 term 是一个 feature。

### Q23. BOW feature space size 是什么？

A：训练/文档集合中的 vocabulary size。

### Q24. BOW 通常应用哪些 preprocessing？

A：tokenisation、stopping、stemming、case folding 等。

### Q25. Word n-grams 特征有什么特点？

A：能捕捉局部词序，但 feature space 更大、更 sparse。

### Q26. Character n-grams 什么时候有用？

A：OCR/ASR degraded text、拼写错误、形态丰富语言等。

### Q27. 句法特征有哪些？

A：POS tags、syntactic tree structures。

### Q28. Topic-based features 有哪些？

A：LDA topics、named entities、links/linked terms。

### Q29. Non-textual features 有哪些？

A：平均文档/句子/词长、大写比例、链接/hashtag/emoji 比例等。

### Q30. Preprocessing 应如何选择？

A：取决于 classification task/application，例如 sentiment 中 punctuation、all caps、emoji 可能很重要。

## Feature Selection and Weighting

### Q31. 为什么需要 feature selection？

A：feature vector 可达 `10^6` 维且稀疏，导致计算成本高和 overfitting。

### Q32. Feature selection 的基本做法是什么？

A：为每个 class 找 top k representative features，再取所有 classes 的 union 构成 reduced feature space。

### Q33. Document frequency 用于 feature selection 有什么问题？

A：会选择 stop words，因为它只看 class 中出现比例。

### Q34. Mutual information 在 feature selection 中衡量什么？

A：term presence/absence 对判断 document 是否属于某 class 的信息量。

### Q35. Chi-squared 在 feature selection 中衡量什么？

A：term appearance 与 class 是否独立，以及偏离独立的程度。

### Q36. Feature weighting 是什么？

A：给 document 中的 feature 赋值，例如 binary、term frequency、TF-IDF、BM25 或 supervised weights。

### Q37. Unsupervised feature weighting 有哪些？

A：TF-IDF、BM25。

### Q38. Supervised feature weighting 有哪些？

A：`tf * MI`、`tf * chi-square` 等。

### Q39. 为什么 scaling 重要？

A：不同特征量纲不同，若不缩放可能让某些特征不合理地主导模型。

## Classifiers and Representations

### Q40. Binary classification 可用哪些 classical classifiers？

A：SVM、random forests、Naive Bayes、k-NN、logistic regression 等。

### Q41. No-free-lunch principle 是什么？

A：没有一种 learning algorithm 在所有 contexts 中都优于其他算法。

### Q42. 文本分类模型实现要适应什么特点？

A：高维、稀疏的文本表示。

### Q43. 哪些算法天然支持 multi-class？

A：decision trees、random forests、Naive Bayes、k-NN、neural networks 等。

### Q44. SVM 做 multi-class 时通常需要什么？

A：组合或级联 binary SVM，例如 one-vs-rest 等策略。

### Q45. Word embeddings 相比 BOW term vectors 有何不同？

A：word embeddings 是 dense vectors，能更直接捕捉语义相似性；BOW vectors sparse 且语义只间接体现。

### Q46. Word embeddings 如何训练？

A：self-supervised 学习，如 CBOW 根据上下文预测中心词，skip-gram 根据目标词预测上下文。

### Q47. Word embeddings 对 OOV 有什么帮助？

A：预训练 embedding 可提供词级语义表示，缓解部分 sparse local representation 问题。

### Q48. Pre-trained language models 如何得到？

A：在大规模未标注语料上做 self-supervised learning，如 next token prediction 或 masked language modelling。

### Q49. Contextualized embeddings 是什么？

A：同一词在不同上下文中可有不同表示，通常来自 BERT/RoBERTa/XLNet 等模型的中间层。

### Q50. Supervised fine-tuning 是什么？

A：先预训练 language model，再用 labeled task data 微调成 classification model。

### Q51. Zero-shot classification 是什么？

A：不提供 task-specific training examples，直接用 language model 按 prompt 分类。

### Q52. Zero-shot classification 的优点是什么？

A：人工成本低，能快速建立 baseline。

### Q53. Zero-shot classification 的缺点是什么？

A：依赖 black-box model，customisation 和 error analysis 空间有限。

### Q54. Few-shot classification 是什么？

A：在 prompt 中给少量 examples，不做参数 fine-tuning，利用 in-context learning 适配任务。

## Evaluation

### Q55. Text classification evaluation 包括哪些维度？

A：effectiveness、efficiency 和 baselines。

### Q56. Effectiveness measures 有哪些？

A：accuracy、precision、recall、F1、per-class metrics、macro-F1 等。

### Q57. Efficiency 可衡量什么？

A：training speed、classification speed、feature extraction speed。

### Q58. 为什么 baseline 重要？

A：判断模型是否真的优于简单策略或现有方法。

### Q59. 常见 baseline 有哪些？

A：random classification、majority class baseline、simple BOW model、LLM zero-shot。

### Q60. Accuracy 什么时候合适？

A：classes 接近平衡且每个 class 同等重要时。

### Q61. Accuracy 在 imbalanced data 中有什么问题？

A：多数类 baseline 也可能 accuracy 很高，但对 minority class 表现很差。

### Q62. Binary imbalanced classification 应关注什么？

A：rare class 的 precision、recall 和 F1。

### Q63. Macro-F1 是什么？

A：先计算每个 class 的 F1，再对 classes 平均，给 minority classes 同等权重。

### Q64. Macro-F1 什么时候重要？

A：classes 不平衡但每个 class 都重要时，例如 gender distribution 80/20。

### Q65. Confusion matrix 有什么用？

A：显示 actual vs predicted classes，帮助找哪些 classes 被混淆。

### Q66. Error analysis 的目标是什么？

A：理解模型错误类型，并据此开发更好 features 或模型。

## Data Splitting and Hyperparameters

### Q67. 为什么要 train/test split？

A：避免 overfitting，并在 unseen data 上评估泛化能力。

### Q68. 常见 train/test split 是什么？

A：例如 80% training、20% test。

### Q69. 为什么需要 development set？

A：用于 hyperparameter optimisation，避免在 test set 上调参。

### Q70. 常见 train/development/test split 是什么？

A：例如 80% training、10% development、10% test。

### Q71. Hyperparameters 的例子有哪些？

A：SVM 的 C、non-linear kernel 的 r/d、binary SVM decision threshold，以及模型选择本身。

### Q72. 为什么在 test data 上调参是 cheating？

A：test set 应只用于最终泛化评估，调参会把模型过拟合到 test set。

## 考前作答模板

### Q73. 如果考“text classification pipeline”，应该怎么答？

A：标注训练样本，提取/选择/加权 features，训练 classifier，在 unseen/test data 上用合适 metrics 评估。

### Q74. 如果考“classification types”，应该怎么答？

A：binary 二分类，SLMC 多类单标签，MLMC 多类多标签；MLMC 常拆成多个 binary tasks。

### Q75. 如果考“feature selection 为什么重要”，应该怎么答？

A：文本 feature space 高维稀疏，feature selection 降低计算成本和 overfitting，可用 DF、MI、chi-square 等。

### Q76. 如果考“evaluation in imbalanced classification”，应该怎么答？

A：accuracy 可能误导，应看 minority/each-class precision、recall、F1 和 macro-F1，并使用 majority baseline 比较。

