# TTDS 01 Introduction 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 TTDS 课程范围、Information Retrieval、Text Classification、Text Analytics、RAG 与课程实践要求。

## 高频核心题

### Q1. TTDS 中的 Text Technologies 主要处理什么对象？

A：主要处理 documents、words、terms 等文本对象，而不是没有文本信息的 images、videos 或 music。

### Q2. TTDS 覆盖哪些核心技术方向？

A：Information Retrieval、Search Engines、Text Classification 和 Text Analytics。

### Q3. Information Retrieval 是否等同于 web search？

A：不是。Web search 只是 IR 的一个重要应用，IR 还包括法律搜索、图书搜索、推荐、过滤、QA、cross-language search 等。

### Q4. IR 的核心定义是什么？

A：IR 是在大规模集合中找到满足用户 information need 的 unstructured material。

### Q5. IR 定义中的四个关键词是什么？

A：Find 是任务，unstructured 是材料性质，information need 是目标，satisfies 需要通过 evaluation 判断。

### Q6. 为什么 IR 不等于简单 Find？

A：Find 通常是 sequential exact match；IR 面对的是不精确、语义模糊、用户需求不完全明确的大规模检索。

### Q7. IR 的典型应用有哪些？

A：Web search、library search、legal search、social search、recommendation、information filtering、speech QA、RAG 检索模块等。

### Q8. 为什么 IR 对 RAG 很重要？

A：RAG 需要先检索相关文档，再让 LLM 抽取、总结或生成答案；检索质量直接影响生成质量。

## Text Classification

### Q9. Text classification 是什么？

A：根据文本内容把 document、article、sentence 等文本输入分类到预定义类别中。

### Q10. Text classification 的输入是什么？

A：文本对象，例如 document、article、sentence、query 或 web page。

### Q11. Text classification 的输出是什么？

A：一个或多个预定义类别。

### Q12. Binary classification 的例子是什么？

A：relevant/irrelevant、spam/not spam、offensive/not offensive。

### Q13. Few-class classification 的例子是什么？

A：sports、politics、comedy、technology 等新闻类别。

### Q14. Hierarchical classification 的例子是什么？

A：专利分类体系中，多层级地给文档分配类别。

### Q15. Text classification 和 IR 有什么联系？

A：二者都处理文本表示和匹配；classification 更关注把文本映射到类别，IR 更关注按 query 找相关文档。

## Course Content

### Q16. 这门课要教你如何构建什么系统？

A：构建 search engine，包括如何 ranking、如何大规模快速检索、如何评价搜索算法。

### Q17. 课程中与文本处理相关的内容有哪些？

A：拼写错误、morphology、synonyms、preprocessing、large text collections 和 text analytics。

### Q18. 课程中与评价相关的内容有哪些？

A：评价 search algorithm、比较 system A 是否真的优于 system B、评价 text classification quality。

### Q19. Text analytics 在本课中主要做什么？

A：分析一组文档与另一组文档的差异，找出主题、词汇或类别层面的区别。

### Q20. 本课提到的重要术语有哪些？

A：inverted index、vector space model、TF-IDF、BM25、language model、PageRank、learning to rank、MAP、MRR、nDCG、mutual information、chi-square、RAG。

### Q21. TTDS 和 ANLP/FNLP 的区别是什么？

A：TTDS 更偏 document-level text technologies、search engines、IR methods/evaluation 和 large-scale text processing，不以 word/phrase-level NLP 为主。

### Q22. TTDS 和 ML practical 的区别是什么？

A：TTDS 会用 off-the-shelf ML 做文本分类，但重点不是一般 ML 理论，而是文本检索、分类和分析系统。

## Practical Orientation

### Q23. 为什么说 TTDS 是 highly practical？

A：课程大量依赖 labs、coursework 和 project，学生会实现搜索引擎和文本技术模块。

### Q24. 课程对数学基础有什么要求？

A：需要 linear algebra、probability、Bayes rule、expectation/variance、calculus、log/exp/ln 等基础。

### Q25. 课程对编程有什么要求？

A：需要 Python、regular expressions、shell commands、data structures 和 software engineering 能力。

### Q26. 为什么 Lab 0 很重要？

A：Lab 0 测试基本文本文件读取、lowercase 和 word count；如果很困难，后续 labs/coursework 会更困难。

### Q27. 课程中 shell commands 为什么重要？

A：处理大规模文本时，cat、sort、grep、uniq、sed 等命令能快速完成数据清洗和统计。

### Q28. Group project 的目标是什么？

A：小组实现一个完整 end-to-end search engine，支持大规模文档搜索和多种文本技术功能。

### Q29. BetterReads 示例说明了什么？

A：课程 project 可以实现真实搜索系统，例如索引 11.5M Goodreads reviews，并综合 relevance 和 sentiment ranking。

## 综合理解题

### Q30. 为什么 IR 同时需要 effectiveness 和 efficiency？

A：系统既要找对相关文档，又要在海量数据和高查询量下快速返回结果。

### Q31. 为什么 web search 不是 IR 的全部，但对 IR 很重要？

A：web search 是规模最大、最常见的 IR 应用，推动了 ranking、indexing、evaluation 和 spam resistance 的发展。

### Q32. 为什么 TTDS 关注 document level 而不是只关注单词级 NLP？

A：搜索、分类、评价和文本分析通常以文档集合为单位，需要处理大规模文档表示与系统效率。

### Q33. RAG 如何体现 TTDS 的价值？

A：RAG 把 IR 作为 LLM 的知识获取模块，说明高质量检索仍是现代生成式 AI 的基础。

## 考前作答模板

### Q34. 如果考“什么是 IR”，应该怎么答？

A：IR 是从大规模、非结构化材料中找到满足用户 information need 的内容，核心问题是 effectiveness 和 efficiency。

### Q35. 如果考“text classification”，应该怎么答？

A：说明输入是文本，输出是预定义类别，可分 binary、multi-class、multi-label 或 hierarchical，并给 spam/news/patent 例子。

### Q36. 如果考“TTDS 的核心技术栈”，应该怎么答？

A：答 indexing、preprocessing、ranked retrieval、evaluation、query expansion、web search、text analytics、classification 和 RAG。

