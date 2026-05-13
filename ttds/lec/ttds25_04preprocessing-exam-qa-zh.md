# TTDS 04 Preprocessing 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 tokenisation、stopping、normalisation、case folding、equivalence classes、stemming、Porter stemmer、preprocessing limitations 和 asymmetric expansion。

## 高频核心题

### Q1. Text preprocessing 的目标是什么？

A：把 document 和 query 中不同表面形式的词转成更容易匹配的 term form，从而提高 retrieval performance。

### Q2. Preprocessing 在 IR pipeline 中位于哪里？

A：在 indexing 前的 text transformation 阶段，同时也应一致地应用于 query processing。

### Q3. 讲义中的标准 preprocessing steps 是什么？

A：Tokenisation、stopping、normalisation 和 stemming。

### Q4. IR 中为什么常说 terms 而不是 words？

A：term 可以是完整单词、词的一部分、数字、代码或经过预处理后的索引单位，不一定是自然语言单词。

### Q5. 为什么 document 和 query 必须用同样 preprocessing？

A：否则 query term 和 indexed term 的形式不一致，会造成匹配失败。

## Tokenisation

### Q6. Tokenisation 是什么？

A：把句子或文本切分成 tokens，每个 token 是一段字符序列。

### Q7. Tokenisation 的典型做法是什么？

A：在 non-letter characters 处分割文本。

### Q8. “Friends, Romans; and Countrymen!” tokenise 后是什么？

A：Friends、Romans、and、Countrymen。

### Q9. Token 和 term 的区别是什么？

A：token 是文本中出现的字符序列实例；term 是进一步处理后用于 index 的单位。

### Q10. Hyphenated words 为什么是 tokenisation 难点？

A：Hewlett-Packard、state-of-the-art、lower-case 可能应作为一个或多个 token，取决于应用需求。

### Q11. Numbers 为什么是 tokenisation 难点？

A：日期、课程代码、电话号码等数字串有特殊结构，简单分割可能破坏意义。

### Q12. URL 为什么是 tokenisation 难点？

A：可以选择整体保留、按 domain 分割或移除，取决于任务。

### Q13. Social media tokenisation 有哪些特殊对象？

A：hashtags、mentions、camel-case hashtags 和不同大小写形式。

### Q14. “San Francisco” 为什么是 tokenisation 难题？

A：它是多词实体，简单按空格切分会丢失整体地名含义。

### Q15. Chinese/Japanese tokenisation 为什么不同？

A：这些语言通常没有空格分词，需要 segmentation。

### Q16. German retrieval 为什么可能需要 compound splitter？

A：德语长复合词包含多个语义成分，拆分可提高匹配，讲义提到可能带来明显性能提升。

## Stopping

### Q17. Stop words 是什么？

A：集合中最常见、通常对 topic aboutness 贡献较少的词，如 the、a、is、to。

### Q18. Stop words 大约占文本多少？

A：讲义中提到约 30-40%。

### Q19. Stop words 是否固定不变？

A：不是。不同 domain 可产生特殊 stop words，例如 tweets 中 RT，patents 中 said/claim。

### Q20. 为什么 stop words 可能仍然重要？

A：在 phrase queries 和 relational queries 中，stop words 可改变意义，例如 “to be or not to be” 或 flights to/from。

### Q21. Web search 为什么倾向保留 stop words？

A：压缩和 query optimisation 降低成本，probabilistic models 会给 stop words 低权重。

### Q22. 如何创建 stop words list？

A：按 term frequency 排序，从最高频 terms 中人工选择可能 stop words。

## Normalisation

### Q23. Normalisation 的目标是什么？

A：让不同 surface forms 映射到相同形式，以提高匹配。

### Q24. Case folding 是什么？

A：把字母统一转成小写，例如 CAR、Car、caR 都变成 car。

### Q25. Case folding 可能有什么风险？

A：Windows 作为操作系统名和 windows 作为普通词可能被合并，丢失专名区别。

### Q26. Diacritics/accents removal 是什么？

A：去除或规范化重音符号，例如 Château -> chateau。

### Q27. Equivalence classes 是什么？

A：把 U.S.A./USA、Ph.D./PhD、多种拼写或标点形式归为等价形式。

### Q28. Equivalence classing 最重要的标准是什么？

A：document 和 query 处理要一致，并尽量遵循用户最常见行为。

## Stemming

### Q29. Stemming 是什么？

A：把词的形态变化还原到共同 stem，以便 play、played、playing 等能匹配。

### Q30. Stemming 主要处理哪些 morphological variations？

A：inflectional variations 如 plurals/tenses，以及 derivational variations 如 verb/noun 转换。

### Q31. Stemming 可以在哪里做？

A：可在 indexing time 做，也可在 query processing 时做。

### Q32. Stemming 对英语 IR 通常提升多少？

A：讲义提到通常约 5-10% effectiveness improvement。

### Q33. Stemming 对高屈折语言为什么更重要？

A：词形变化更多，同一词根有许多 surface forms；讲义提到 Finnish 和 Arabic 中提升更大。

### Q34. Dictionary-based stemmer 是什么？

A：使用词表或相关词列表识别同源/相关词。

### Q35. Algorithmic stemmer 是什么？

A：使用规则程序移除后缀或转换形式，例如 suffix-s stemmer 和 Porter stemmer。

### Q36. Algorithmic stemming 的 false negative 是什么？

A：应合并但没有正确合并，例如 supplies -> supplie。

### Q37. Algorithmic stemming 的 false positive 是什么？

A：不应合并却被错误改变，例如 James -> Jame。

### Q38. Porter stemmer 的特点是什么？

A：英语最常用 stemmer，使用 conventions 和 5 phases of reductions 顺序应用规则。

### Q39. Stemmed terms 为什么可能像 misspelled words？

A：stem 不是给用户看的自然词，而是 IR 系统内部用于匹配的 term form，如 repli、replac。

## Limitations 与 Expansion

### Q40. Preprocessing 无法解决哪些问题？

A：irregular verbs、spelling variants、abbreviations 和 synonyms，如 went/go、colour/color、TV/television、car/vehicle。

### Q41. Query expansion 解决什么问题？

A：通过扩展 query 加入相关词或同义词，处理 preprocessing 无法统一的语义差异。

### Q42. Asymmetric expansion 是什么？

A：不把词完全归为同一 equivalence class，而是维护非对称匹配关系，例如 query window 可搜 window/windows，但 Windows 可能保持专名。

### Q43. Asymmetric expansion 的优点是什么？

A：更灵活，能保留大小写或专名差异。

### Q44. Asymmetric expansion 的缺点是什么？

A：词表更大、query 更长、效率更低，且 term statistics 可能不准确。

### Q45. Common preprocessing practice 是什么？

A：按 non-letter 分词，移除 stop words，case folding，应用 Porter stemmer；特殊任务如 tweets 需保留 # 和 @。

## 考前作答模板

### Q46. 如果考“preprocessing pipeline”，应该怎么答？

A：说明 tokenisation 把文本切为 tokens，stopping 去掉低信息高频词，normalisation 合并表面形式，stemming 合并形态变化，并强调 document/query 必须一致。

### Q47. 如果考“stop words 是否总该删除”，应该怎么答？

A：不是。它们对 topic 贡献小但对 phrase/relational queries 很重要；web search 常保留，因为压缩、优化和低权重处理成本较小。

### Q48. 如果考“stemming 的利弊”，应该怎么答？

A：利是合并形态变化、提升 recall/effectiveness；弊是 false positives/negatives，且生成的 stems 不一定是合法单词。

