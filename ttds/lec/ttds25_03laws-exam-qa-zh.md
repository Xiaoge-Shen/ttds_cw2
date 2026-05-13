# TTDS 03 Laws of Text 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 Zipf's law、Benford's law、Heaps' law、clumping/contagion，以及这些规律对 indexing 和 text processing 的意义。

## 高频核心题

### Q1. 为什么要学习 text laws？

A：文本中的词频、词汇增长和词项出现位置具有稳定统计规律，可用于估计 index size、vocabulary size 和处理策略。

### Q2. Text laws 适用于哪些范围？

A：不同语言、不同文本领域和大规模文本集合中都常能观察到类似规律。

### Q3. 为什么 word 是表示文本的基本单位？

A：IR 通常用词项作为 document/query 的 feature，词频和分布决定检索和索引行为。

### Q4. 高频词和低频词在文本中如何分布？

A：少数词非常频繁，大量词很少出现，词频呈 hard exponential decay。

### Q5. 讲义中提到多少比例的 terms 可能只出现一次？

A：约 50% terms appear once。

## Zipf's Law

### Q6. Zipf's law 描述什么？

A：在文本集合中，按词频排序后，term rank 与其出现概率/频率大致成反比。

### Q7. Zipf's law 的基本公式是什么？

A：`r * P_t ≈ constant`，也可写成 `r * freq_t ≈ constant`。

### Q8. Zipf's law 中 `r` 表示什么？

A：term 按 frequency 排序后的 rank。

### Q9. Zipf's law 中 `P_t` 表示什么？

A：term `t` 在集合中出现的概率。

### Q10. Zipf's law 对 stop words 有什么解释？

A：少数 stop words 如 the、of、to 占据最高频 rank，频率远高于普通词。

### Q11. Zipf's law 对 rare words 有什么解释？

A：低频词数量巨大，很多词只出现一次或少数几次。

### Q12. Zipf's law 对 inverted index 有什么影响？

A：少数高频词 postings list 很长，大量低频词 postings list 很短，index 极度稀疏。

### Q13. 为什么 Zipf's law 会影响 query processing？

A：高频词 postings list 长，处理代价高；低频词更有判别力，常更适合先处理。

## Benford's Law

### Q14. Benford's law 描述什么？

A：许多自然数据中的数字首位并非均匀分布，而是小数字如 1 更常见。

### Q15. Benford's law 的公式是什么？

A：`P(d) = log(1 + 1/d)`，其中 `d` 是首位数字。

### Q16. 哪些数据可能服从 Benford-like law？

A：term frequencies、physical constants、energy bills、population numbers 等。

### Q17. Benford's law 和 Zipf's law 的关系是什么？

A：都反映自然数据中的非均匀、长尾分布，词频首位也可能呈 Benford-like pattern。

## Heaps' Law

### Q18. Heaps' law 描述什么？

A：随着读入文本总词数增加，observed vocabulary size 会继续增长但增长速度逐渐下降。

### Q19. Heaps' law 的公式是什么？

A：`v(n) = k * n^b`，其中 `v` 是 unique words 数，`n` 是总 words 数。

### Q20. Heaps' law 中 `b` 的典型范围是什么？

A：通常 `0.4 < b < 0.7`，且 `b < 1`。

### Q21. Heaps' law 中 `k` 表示什么？

A：与 collection 和 preprocessing 有关的常数，影响 vocabulary growth 的规模。

### Q22. 为什么 Heaps' law 不会很快 saturate？

A：真实大型集合不断出现新实体、拼写、专业词和噪声，vocabulary 会持续增长。

### Q23. Heaps' law 对 index size 有什么意义？

A：可用 collection size 估计 vocabulary size，从而估计 dictionary 和 index 规模。

### Q24. 20 billion terms、`k=0.25`、`b=0.7` 时 unique terms 大约多少？

A：讲义估计约 4 million unique terms。

### Q25. Heaps' law 在小 collection 上是否准确？

A：不一定。讲义指出当 `n` 很小时不太准确。

## Clumping/Contagion

### Q26. Clumping/contagion in text 是什么？

A：词项虽是稀有事件，但一旦出现一次，往往会在附近再次出现。

### Q27. 为什么词像 rare contagious disease，而不像 independent lightning？

A：因为主题局部性强，某个话题出现后相关词会在同一段或同一文档附近聚集。

### Q28. 如何观察 clumping？

A：找只出现两次的 terms，测两次 occurrence 的距离，通常发现多数距离较近。

### Q29. Clumping 对 retrieval 有什么启发？

A：词的局部聚集能帮助判断文档主题，也解释了 passage/proximity 信息为什么有用。

### Q30. Clumping 对 compression/indexing 有什么影响？

A：term occurrences 可能在局部密集，有助于设计 positional/proximity index 和压缩策略。

## Shell/Practical

### Q31. 本讲涉及哪些 shell commands？

A：cat、zcat/gzcat、more、tr、sort、uniq、redirection `>` 和 pipe `|` 等。

### Q32. 为什么 sort 和 uniq 对 text law 实验有用？

A：可统计 term frequencies，从而观察 Zipf、singleton terms 和 vocabulary growth。

### Q33. 为什么实践中要对另一种语言也尝试 text laws？

A：这些规律跨语言存在，但参数和具体分布会随语言、领域、preprocessing 改变。

## 考前作答模板

### Q34. 如果考“Zipf's law”，应该怎么答？

A：定义为 term rank 与 frequency/probability 近似反比，公式 `r * P_t ≈ constant`，说明少数高频词和大量低频词造成 index 稀疏。

### Q35. 如果考“Heaps' law”，应该怎么答？

A：写 `v(n)=k n^b`，说明 vocabulary 随 collection size 增长但速度变慢，可用于估计 unique terms 和 index dictionary size。

### Q36. 如果考“contagion”，应该怎么答？

A：说明词出现具有局部聚集性，一旦某词出现，更可能在附近再次出现；这与主题局部性和 proximity search 有关。

