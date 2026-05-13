# TTDS 14 Web Search 1 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 web collection 特性、massive data、web graph、PageRank、random surfer、anchor text、link spam 和 PageRank 的现实局限。

## 高频核心题

### Q1. Web document collection 有什么特点？

A：规模巨大、无统一设计或协调、持续增长、质量参差不齐，并存在 spam/fake/generated content。

### Q2. Web search 的技术挑战有哪些？

A：storage、processing、networking、crawling、indexing、quality control 和 spam resistance。

### Q3. Web search 中 relevance 的特殊挑战是什么？

A：query 语义模糊、用户主观性强，且 SEO/spam 会故意操纵排名。

### Q4. Massive data 对 web search 有什么好处？

A：更大的 index 往往包含更多 relevant docs，使 top-k precision 可能提升。

### Q5. 为什么 larger index 可能胜过 better algorithm？

A：在 web search 中，更多数据可能让系统在 top ranks 找到更多相关文档，即使算法改进有限。

### Q6. “Big data or clever algorithm” 的启发是什么？

A：在某些任务中，大数据覆盖能弥补算法简单性，但最终最好结合大数据和好算法。

## Web Graph

### Q7. Web 可如何表示为 directed graph？

A：web pages 是 nodes，hyperlinks 是 directed edges。

### Q8. Link assumption 1 是什么？

A：从 page A 到 page B 的 hyperlink 表示作者认为 B 有某种 relevance/quality。

### Q9. Link assumption 2 是什么？

A：anchor text 描述 target page 的内容。

### Q10. Google 对 PageRank 的基本解释是什么？

A：把从 A 到 B 的 link 看成 A 给 B 的 vote。

### Q11. 为什么不是所有 votes 等价？

A：来自 high-quality/high-PageRank page 的 link 比普通 blog link 更重要。

### Q12. PageRank 是 content-dependent 还是 content-independent？

A：主要是 content-independent 的 quality/authority signal。

### Q13. PageRank 如何用于 ranking？

A：作为 ranking feature 与 content relevance 等其他 features 结合。

## PageRank

### Q14. Random surfer model 是什么？

A：用户从随机 page 开始，不断随机点击 outgoing links，并以一定概率跳到随机 page。

### Q15. PageRank of page x 是什么？

A：random surfer 在随机时刻位于 page x 的概率。

### Q16. 为什么 random surfer 需要 jump/teleport？

A：避免陷入 dead ends 或 cycles，例如无 outlinks page 或 B<->C 循环。

### Q17. PageRank 初始化通常怎么做？

A：对 N 个 pages，初始 `PR0(x)=100%/N` 或等分概率。

### Q18. PageRank 更新公式的核心是什么？

A：`PR_{i+1}(x) = (1-λ)/N + λ * sum_{y->x} PR_i(y)/L_out(y)`。

### Q19. `λ` 表示什么？

A：继续沿 hyperlink 浏览的概率；`1-λ` 是随机跳转概率。

### Q20. `L_out(y)` 表示什么？

A：page y 的 outgoing links 数量。

### Q21. 一个 page 对 x 的贡献是什么？

A：如果 `y -> x`，则 y 把自己的 PageRank 平均分给 outgoing links，贡献 `PR(y)/L_out(y)`。

### Q22. PageRank scores 应满足什么总和？

A：迭代收敛后总概率应为 100% 或 1。

### Q23. 没有 inlinks 的 page PageRank 是什么？

A：只得到 teleportation 部分，例如 `(1-λ)/N`。

### Q24. 为什么一个 high-PR inlink 可能胜过多个 low-PR inlinks？

A：PageRank 递归衡量 link source 的重要性，高质量 source 的 vote 更值钱。

### Q25. 对称 inlinks 的 pages 会怎样？

A：若链接结构对称，可能获得相同 PageRank。

## Anchor Text

### Q26. Anchor text 是什么？

A：hyperlink 中可见的链接文本，用来描述 target page。

### Q27. Anchor text 为什么像 human query expansion？

A：不同网页作者会用不同短语描述同一 target page，形成多样化描述。

### Q28. Anchor text 如何用于 indexing？

A：把所有指向某 page 的 anchor texts 加入该 page 的 index representation。

### Q29. Anchor text 是否可加权？

A：可以，根据 linking page 的 PageRank 或质量给不同 anchor text 不同权重。

### Q30. Anchor text 为什么能显著提升 retrieval？

A：它提供 target page 自身可能没有的 concise descriptive terms。

## Vulnerabilities 与 Spam

### Q31. Hawthorne Effect 是什么？

A：被观察会改变行为；ranking signals 被公开后，网站会调整行为。

### Q32. Goodhart's Law 是什么？

A：当一个 measure 成为 target，它就不再是好 measure。

### Q33. Cobra Effect 是什么？

A：激励设计不当导致解决方案反而让问题变糟。

### Q34. Trackback link spam 是什么？

A：利用 “blogs that link to me” 等机制制造人工 feedback loops 和垃圾链接。

### Q35. Comment spam 是什么？

A：在高 PageRank 网站评论中放 spam links，试图转移 PageRank。

### Q36. `rel=nofollow` 的作用是什么？

A：告诉 search engine 不把该 link 计入 PageRank 等 link signal。

### Q37. Link farm 是什么？

A：大量 fake densely-connected pages/domains 构成的 clique，用来操纵 link-based ranking。

### Q38. Anchor text spam 是什么？

A：通过大量 anchor text 操纵某页面对某 query 的排名。

### Q39. 为什么 high PageRank websites 可能 gaming algorithm？

A：它们可能避免 outlinks、制造 irrelevant content 或利用自身 authority 获取流量。

## Reality

### Q40. PageRank 是否仍是 Google ranking 的全部？

A：不是。它曾是重大突破，但现在只是众多 ranking features 之一。

### Q41. 现代 web ranking 还使用什么？

A：content features、user behavior、spam signals、machine-learned ranking、Learning to Rank 等。

### Q42. PageRank 的核心价值是什么？

A：利用 web graph 的 collective linking behavior 估计 page authority。

## 考前作答模板

### Q43. 如果考“PageRank”，应该怎么答？

A：定义为 random surfer 在某 page 的稳态概率，公式包含 teleportation 和 incoming links contribution；high-PR pages 的 links 更有价值。

### Q44. 如果考“anchor text”，应该怎么答？

A：说明 anchor text 是 link text，常是 target page 的短描述，可作为 human query expansion 加入 target page index，提升 retrieval。

### Q45. 如果考“PageRank vulnerability”，应该怎么答？

A：说明当 link/anchor 成为 ranking target，会产生 comment spam、trackback spam、link farms 和 anchor text spam，因此 PageRank 只能作为众多特征之一。

