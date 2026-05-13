# TTDS 02 Definitions 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 IR 基本概念：documents、queries、information need、relevance、bag-of-words、IR vs database，以及 indexing/search architecture。

## 高频核心题

### Q1. IR 的基本形式是什么？

A：给定 query Q，从 document collection 中找到 relevant documents D。

### Q2. IR 的两个主要问题是什么？

A：Effectiveness 和 efficiency。

### Q3. Effectiveness 在 IR 中是什么意思？

A：系统能否找到真正相关、能满足用户 information need 的文档。

### Q4. Efficiency 在 IR 中是什么意思？

A：系统能否在大规模数据和高查询负载下快速完成检索。

### Q5. 为什么 IR 和 relational database 不同？

A：IR 多处理 unstructured free text，query 模糊，结果不精确；数据库处理 structured data、formal query，结果通常 exact。

### Q6. IR main components 有哪些？

A：Documents、queries 和 relevant documents。

## Documents 与 Queries

### Q7. Document 在 IR 中是什么？

A：被检索的基本单位，通常有 unique ID，N 个 documents 构成 collection。

### Q8. Document 一定是传统文本文件吗？

A：不一定，可以是 web page、email、book、sentence、tweet、product description、advertisement、code，甚至无词对象如 DNA。

### Q9. Query 是什么？

A：用户用来表达 information need 的输入，常见形式是 free text。

### Q10. 同一 information need 是否只能由一个 query 表达？

A：不是。同一需求可用多个 query 表达，例如 hurricane news、North Carolina storm、Florence。

### Q11. 同一 query 是否只能表达一个需求？

A：不是。Apple 可指公司或水果，Jaguar 可指动物、汽车品牌等。

### Q12. Query 可以有哪些形式？

A：web keywords、image sample、QA question、humming tune、user history、structured scholar search、advanced search operators。

### Q13. 为什么 query formulation 是 IR 的难点？

A：用户常用短 query 表达复杂需求，而且存在 synonymy、polysemy 和上下文缺失。

## Relevance

### Q14. Relevance 的核心问题是什么？

A：判断 document D 是否匹配 query Q 或能否满足用户 information need。

### Q15. Relevance 为什么 tricky？

A：它可能取决于用户是否喜欢、是否点击、是否能完成任务、是否 novel，以及主题是否匹配。

### Q16. Relevance 的主题视角是什么？

A：D 和 Q 是否 share similar meaning，是否 about the same topic/subject/issue。

### Q17. Synonymy 如何影响 relevance？

A：不同词可表示相同意思，例如 Edinburgh festival 和 The Fringe，导致词面不匹配但语义相关。

### Q18. Polysemy 如何影响 relevance？

A：同一词有多个含义，例如 Apple 或 Jaguar，导致 query ambiguous。

### Q19. Web relevance 还面临什么特殊问题？

A：SEO、spam 和 adversarial content 会干扰检索质量。

### Q20. “Relevant items are similar” 的基本假设是什么？

A：使用相似 vocabulary 的 documents 往往具有相似 meaning，因此可用 overlap 或 probabilistic retrieval model 判断匹配。

## Bag-of-Words

### Q21. Bag-of-words 是什么？

A：把 document/query 看成带重复的词袋，忽略原始词序，用词项及其重复次数表示文本。

### Q22. 为什么 BOW 在 IR 中有效？

A：重排文本通常不会完全破坏 aboutness，单词可作为主题含义的 building blocks。

### Q23. BOW 中 match 通常意味着什么？

A：document 和 query 之间的词项 overlap 或基于词项特征的相似度。

### Q24. BOW 的主要优点是什么？

A：简单、可扩展、使 retrieval models tractable。

### Q25. BOW 的主要批评是什么？

A：忽略词序、句法和部分上下文，对 negation、多词表达和无空格语言有困难。

### Q26. BOW 是否完全丢弃 context？

A：不完全。它主要丢弃 surface form 和 well-formedness，但主题词仍保留大量 aboutness 信息。

### Q27. 对 Chinese/Japanese 这类无空格语言，BOW 需要什么？

A：需要 segmentation 或 feature induction，把文本切分为能反映 aboutness 的单位。

### Q28. 对 German compounds，BOW 可能需要什么？

A：可能需要 decompounding，把复合词拆成可匹配的语义单位。

## IR System Architecture

### Q29. IR black box 中有哪些函数？

A：query representation function、document transformation function、comparison function、retrieval model 和 index。

### Q30. Indexing process 是什么？

A：离线过程，把数据获取、存储、文本转换为 BOW/terms，并创建 index。

### Q31. Search process 是什么？

A：在线过程，帮助用户构造 query、检索并排序结果、展示结果、记录用户行为并迭代优化。

### Q32. 为什么 indexing 是 offline，而 search 是 online？

A：indexing 可预先完成以加速查询；search 必须响应实时用户请求。

### Q33. Search process 中 log data 有什么用？

A：记录 clicks、hovering、giving up 等行为，用于评价、优化 ranking 和改进系统。

### Q34. RAG 和 IR 在架构上有什么关系？

A：RAG 中 search engine 检索 relevant documents，然后 LLM extract/summarise 或生成答案。

## 综合比较题

### Q35. IR 和 DB 的 retrieval/data/query/result 有何区别？

A：DB 检索结构化数据、query formal and unambiguous、结果 exact；IR 检索 unstructured free text、query natural language、结果需要 relevance evaluation。

### Q36. Effectiveness 和 efficiency 为什么经常 trade off？

A：更复杂模型可能提升相关性但增加计算成本；更快索引结构可能牺牲部分匹配细节。

### Q37. 为什么 interaction 对 IR 很重要？

A：用户需求可能不清晰，系统可通过 query suggestion、结果浏览、reformulation 和 logs 逐步满足需求。

## 考前作答模板

### Q38. 如果考“relevance 为什么困难”，应该怎么答？

A：提 synonymy、polysemy、subjectivity、novelty、task satisfaction 和 web spam，并说明 relevance 不等于 exact string match。

### Q39. 如果考“bag-of-words”，应该怎么答？

A：定义为忽略词序的词袋表示；优点是简单可扩展，缺点是丢失词序/句法，对 negation 和某些语言需要额外处理。

### Q40. 如果考“IR architecture”，应该怎么答？

A：说明 offline indexing 获取并转换文档、建立 index；online search 处理 query、ranking、展示结果并记录交互。

