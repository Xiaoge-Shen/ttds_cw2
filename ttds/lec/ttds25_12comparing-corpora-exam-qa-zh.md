# TTDS 12 Comparing Text Corpora 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 content analysis、word frequency analysis、word-level differences、mutual information、chi-squared、dictionaries/lexicons、dominance scores 和 topic modelling 入门。

## 高频核心题

### Q1. Comparing text corpora 的基本场景是什么？

A：给定两个或多个包含大量 plain text files 的 corpora，想系统理解内容是什么，以及不同 corpora 如何不同。

### Q2. 初步文本分析可以先做什么？

A：读样例、做 language identification、统计词数/高频词/平均文档长度、画 word clouds 等。

### Q3. 为什么不能只“ask AI”？

A：AI 总结可能不可复现、不系统、缺少量化依据，且难以作为严格 corpus comparison 的统计证据。

### Q4. Content analysis 的目标是什么？

A：确定文档中有哪些 content types/themes/topics，以及哪些文档包含哪些 topics。

### Q5. 传统 content analysis 的步骤是什么？

A：读子集定义 themes，制定 coding methodology，标注所有文档，检查 coder agreement，调解分歧，分析 annotations。

### Q6. Automated content analysis 为什么有用？

A：人类更准确但慢，机器可处理海量文本，适合现代大规模 corpora。

### Q7. Automated content analysis 对单一 corpus 可做什么？

A：word frequency analysis、dictionaries/lexicons、topic modelling。

### Q8. 对多个 corpora/classes 可做什么？

A：word-level differences、dominance scores、topic-level differences。

## Word-level Analysis

### Q9. Word frequency analysis 的基本步骤是什么？

A：preprocess 文本，count words，按 document length normalize，再 across documents 平均。

### Q10. 为什么 word frequency 要 normalize by document length？

A：否则长文档会因词数多而主导频率统计。

### Q11. Word-level differences 想回答什么？

A：哪些 words 最能 characterize 某个 corpus 或 class。

### Q12. 为什么 word-level comparison 需要 reference corpus？

A：只有与背景比较，才能判断某词在目标 corpus 中是否特别突出。

### Q13. Word-level difference 的方法有哪些？

A：Mutual information 和 chi-squared 等。

### Q14. 这些方法还可用于什么？

A：feature selection，选择最能区分类别的 terms。

## Mutual Information

### Q15. Mutual information `I(X;Y)` 衡量什么？

A：观察变量 X 后能获得多少关于 Y 的信息。

### Q16. Mutual information 和 information gain 的关系是什么？

A：在这里讲义指出二者是同一概念。

### Q17. Mutual information 是否等于 pointwise mutual information？

A：不是。MI 是整体变量间信息量，PMI 是具体事件对的关联。

### Q18. 在 word-class 分析中 X 和 Y 可以是什么？

A：`X=U` 表示 document 是否包含 term `t`，`Y=C` 表示 document 是否属于 target class。

### Q19. MI 在 corpus comparison 中如何解释？

A：term presence/absence 对判断文档是否属于某 class 的帮助越大，MI 越高。

### Q20. MI 的优点是什么？

A：能量化 term 与 class 之间的信息关系，适合选 class-characteristic terms。

### Q21. MI 的潜在问题是什么？

A：对稀有但偶然与 class 共现的 terms 可能敏感，需要结合频率和解释性判断。

## Chi-squared

### Q22. Chi-squared 是什么类型方法？

A：hypothesis testing approach。

### Q23. Chi-squared 的 null hypothesis 是什么？

A：term appearance 与 document class 独立。

### Q24. 在 corpus comparison 中 chi-squared 高说明什么？

A：term 出现与 class membership 之间偏离独立性较强。

### Q25. Chi-squared 和 MI 的共同用途是什么？

A：识别区分类别或 corpus 的重要 words，并可用于 feature selection。

### Q26. Chi-squared 更常用于什么？

A：讲义提到它更多用于 classes 之间比较。

## Dictionaries and Lexicons

### Q27. Dictionary/lexicon approach 是什么？

A：使用预先构建的 category -> word list 映射，统计文档中某类词的比例或分数。

### Q28. 一个简单 sentiment lexicon 例子是什么？

A：Positive 包含 good/great/happy，Negative 包含 terrible/bad/worst 等。

### Q29. 为什么 domain 对 dictionary 很重要？

A：“unpredictable movie plot” 可是正面，“unpredictable coffee pot” 则可能负面。

### Q30. Lexicon category score 如何计算？

A：`num_dictionary_words_in_document / num_total_words_in_document`。

### Q31. Dictionary scores 还能怎么用？

A：作为 machine learning features。

### Q32. 常见 dictionaries 有哪些？

A：LIWC、General Inquirer、VADER、SentiWordNet、EmoLex、Empath 等。

### Q33. Dominance score 是什么？

A：某 category 在 target corpus 中的 score 与 background corpus 中 score 的比值。

### Q34. Dominance score 用来回答什么？

A：某类词或主题在目标 corpus 中是否相对更突出。

## Topic-level Analysis

### Q35. Topic modelling 的目标是什么？

A：发现 corpus 中主要 themes/topics，并估计哪些 documents 包含哪些 topics。

### Q36. Topic model 的输入输出是什么？

A：输入 corpus；输出 document-topic matrix 和 topic-word matrix。

### Q37. Document-topic matrix 表示什么？

A：每个 document 中各 topics 的比例或权重。

### Q38. Topic-word matrix 表示什么？

A：每个 topic 中 words 的概率或权重分布。

### Q39. Topic modelling 如何看作 dimensionality reduction？

A：把高维 word space 压缩为低维 topic space。

### Q40. Topic models 是否只用于 text？

A：不是，也可用于 bioinformatics、code、music、network data 等。

### Q41. 最流行的 topic modelling 方法是什么？

A：Latent Dirichlet Allocation (LDA)。

### Q42. 其他 topic modelling 方法有哪些？

A：pLSI、PCA-based methods、non-negative matrix factorization、deep learning based topic modelling。

## 考前作答模板

### Q43. 如果考“automated content analysis”，应该怎么答？

A：说明它把人工 content analysis 的 themes/topics/labels 过程用 word frequencies、lexicons、topic models 或 classifiers 自动化，以处理大规模 corpora。

### Q44. 如果考“MI/chi-squared 的作用”，应该怎么答？

A：它们量化 term 与 class/corpus 的关联，可用于找 characteristic words 和 feature selection；chi-squared 检验 term 与 class 是否独立。

### Q45. 如果考“lexicon approach”，应该怎么答？

A：用 category-to-word-list 统计文档或 corpus 中类别词比例，优点简单可解释，缺点是 domain-sensitive、覆盖有限。

