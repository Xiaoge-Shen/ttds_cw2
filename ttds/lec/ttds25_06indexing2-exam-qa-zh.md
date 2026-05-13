# TTDS 06 Indexing 2 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 structured documents、extent index、index compression、delta/v-byte encoding、dictionary data structures、wild-card queries、permuterm index、character n-gram index 和 spelling correction。

## 高频核心题

### Q1. Structured documents 有哪些结构信息？

A：metadata、title、author、timestamp、headline、section、body、tags、links、hashtags、mentions 等。

### Q2. 处理 structured documents 有哪些方案？

A：忽略结构、为每个 field 建 separate index，或使用 extent index。

### Q3. Extent index 是什么？

A：把 field/tag/element 当作特殊 term，同时记录某个 field 在文档中的 span 范围。

### Q4. Extent index 能支持什么查询？

A：field-specific queries，例如 `Headline: lecture`，即只在 headline span 中查找 term。

### Q5. Extent posting 通常存什么？

A：field/tag 在 document 中的 span，例如 `docID, start:end`。

### Q6. Extent index 的优点是什么？

A：允许结构化字段查询，并支持 overlapping spans。

## Index Compression

### Q7. 为什么 inverted indices 很大？

A：大规模 collection 中有大量 terms、docIDs、positions 和 frequencies，需要大量 disk space 与 I/O。

### Q8. Index compression 的目标是什么？

A：减少 storage 和 I/O，让更多 index chunks 可缓存到 memory 中。

### Q9. Index compression 为什么可能让系统更快？

A：虽然 decompression 增加 processing，但节省 I/O 往往更大，整体更快。

### Q10. Delta encoding 是什么？

A：不存完整 docIDs，而存相邻 docID 的差值 gaps。

### Q11. Delta encoding 为什么有效？

A：postings list 中 docIDs 递增，gaps 通常比原 docID 小，尤其高频词中 gaps 很小。

### Q12. Delta encoding 对 frequent terms 为什么特别有用？

A：频繁词出现在很多文档中，相邻 docIDs 差值小，压缩收益大。

### Q13. v-byte encoding 是什么？

A：用可变字节数存整数，高位标记 terminate/continue，剩余 7 bits 存数字。

### Q14. v-byte encoding 为什么适合 gaps？

A：小 gaps 可用更少字节表示，大 gaps 才用更多字节。

### Q15. 更高压缩率的代价是什么？

A：更少 storage/I/O，但需要更多 processing/decompression。

### Q16. Elias gamma code 是什么？

A：一种比 v-byte 更复杂的整数压缩编码方法，讲义作为更 sophisticated compression 例子提到。

## Dictionary Data Structures

### Q17. Dictionary data structure 存什么？

A：term vocabulary、document frequency、指向 postings list 的 pointers 等。

### Q18. 为什么不能把所有 index 都 load 到 memory？

A：真实 collection 太大，完整 index 可能超过 memory，只能缓存或载入关键结构。

### Q19. Hash dictionary 的优点是什么？

A：term lookup 快，平均可达 `O(1)`。

### Q20. Hash dictionary 的缺点是什么？

A：不易支持 minor variants、prefix search，vocabulary 增长时可能需要 rehash。

### Q21. Tree dictionary 的优点是什么？

A：支持 prefix search 和词典范围查询。

### Q22. Tree dictionary 的缺点是什么？

A：lookup 较慢，通常 `O(log M)`，且需要平衡维护。

### Q23. B-tree 为什么常用于索引？

A：它减轻 rebalancing 问题，适合磁盘/大规模 dictionary 存储和范围查询。

## Wild-card Queries

### Q24. Wild-card query `mon*` 表示什么？

A：查找所有以 mon 开头的 terms。

### Q25. 为什么 `mon*` 对 B-tree 容易？

A：它是 prefix search，可在排序词典中找到 mon 开头的连续范围。

### Q26. 为什么 `*mon` 更难？

A：它是 suffix search；可通过维护 reversed terms 的 B-tree 支持。

### Q27. Query `se*ate AND fil*er` 为什么昂贵？

A：每个 wildcard 可能扩展成许多 terms，形成大的 disjunction，再做 postings 操作。

### Q28. Permuterm index 的核心思想是什么？

A：为每个 term 加 `$` 并存所有 rotations，使 wildcard 可旋转成 prefix query。

### Q29. Term `hello` 的 permuterm rotations 有哪些？

A：`hello$`、`ello$h`、`llo$he`、`lo$hel`、`o$hell`、`$hello`。

### Q30. Permuterm 如何处理 `he*lo`？

A：把 wildcard 旋到末尾，变成类似 `lo$he*` 的 prefix lookup。

### Q31. Permuterm index 的主要缺点是什么？

A：index size 很大，因为每个 term 要存多个 rotations。

## Character n-gram Index

### Q32. Character n-gram index 是什么？

A：把词拆成字符 n-grams，并建立 n-gram 到 dictionary terms 的二级 index。

### Q33. 为什么使用 `$` 边界符号？

A：标记 word boundary，使前缀/后缀信息可被 n-grams 表示。

### Q34. Character n-gram index 通常是几层？

A：两层：character n-grams -> possible terms；terms -> documents。

### Q35. Query `mon*` 用 bigram index 如何处理？

A：先查 `$m AND mo AND on` 得到候选 terms，再 post-filter 满足 `mon*` 的 terms。

### Q36. 为什么 n-gram query 后必须 post-filter？

A：n-gram 匹配可能返回 false positives，例如 `$m AND mo AND on` 也可能匹配 moon。

### Q37. Character n-gram wildcard search 的最终步骤是什么？

A：把 surviving terms 在 normal term-document index 中查找并 OR 组合结果。

### Q38. Wild-card 查询为什么可能很贵？

A：候选 terms 多，最终会形成很大的 disjunction。

## Applications

### Q39. Character n-gram index 如何用于 spelling correction？

A：把 dictionary words 当作 documents，character n-grams 当作 terms；misspelled query 查找 n-gram overlap 最大的候选词。

### Q40. `elepgant` spelling correction 的直觉是什么？

A：它与 elegant、elephant 等词共享很多 character n-grams，因此可作为候选 correction。

### Q41. 哪些场景可直接用 char n-grams 作为 index terms？

A：无可靠 stemmer/segmenter 的语言、OCR documents、ASR errors、拼写错误较多的文本。

### Q42. 多个 n 的 char representation 有什么好处？

A：同时捕捉短字符片段和更长局部结构，提高鲁棒性。

## 考前作答模板

### Q43. 如果考“extent index”，应该怎么答？

A：说明它把字段/标签作为特殊 term，并记录字段 span，支持 field-specific search 和 overlapping structures。

### Q44. 如果考“delta + v-byte compression”，应该怎么答？

A：先用 delta encoding 把 docIDs 变成 gaps，再用 v-byte 以可变字节数存小整数，减少 I/O。

### Q45. 如果考“wild-card search 方法比较”，应该怎么答？

A：B-tree 支持 prefix，reversed tree 支持 suffix，permuterm 支持一般 wildcard 但 index 大，char n-gram 更灵活但需 post-filter 且 query 可能昂贵。

