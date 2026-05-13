# TTDS 15 Web Search 2 考试复习 QA

这份是一套面向考试的整讲复习问答，不按页码整理。重点覆盖 web search history、sponsored search、CPC/CPM/RPM、user intent、SEO、cloaking、duplicate detection、shingles/MinHash、web crawling、crawler architecture 和 politeness。

## 高频核心题

### Q1. 早期 web search engines 使用什么技术？

A：1995-1997 年的 Altavista、Excite、Infoseek、Lycos、AOL 等主要依赖 keyword-based traditional IR techniques。

### Q2. 早期 search engines 的主要问题是什么？

A：scalability 和 ranking quality。

### Q3. Goto/Overture 的 paid ranking 是什么？

A：search ranking 取决于广告主愿意为关键词支付多少钱，即 sponsored search。

### Q4. CPC 是什么？

A：Cost Per Click，按点击付费。

### Q5. CPM 是什么？

A：Cost Per Thousand Impressions，按千次展示付费。

### Q6. RPM 是什么？

A：Revenue per 1000 video views，常用于视频平台收入分析。

### Q7. Google 1998 后的关键突破是什么？

A：link-based ranking，尤其 PageRank。

### Q8. Google 如何结合 paid search 和 algorithmic search？

A：把 paid search ads 与 algorithmic search results 分开展示，广告不直接等同自然排名。

### Q9. 2024+ web search 的新趋势是什么？

A：AI + RAG systems，但尚无明确最终赢家。

## Web Search Basics

### Q10. Web search engine 的基本组件有哪些？

A：web spider/crawler、indexer、indexes/ad indexes 和 search/ranking component。

### Q11. Web spider 做什么？

A：抓取网页、解析链接、把页面和 URLs 交给 indexing/crawling pipeline。

### Q12. Indexer 做什么？

A：处理 crawled pages，建立 searchable indexes。

### Q13. Algorithmic search results 和 sponsored ads 有何区别？

A：algorithmic results 基于 relevance/quality ranking；sponsored ads 基于广告系统和竞价等。

### Q14. Sponsored results 为什么可能优先显示？

A：商业广告位可能在页面上更显眼或更靠前。

## User Intent

### Q15. Web search user needs 有哪三类？

A：Informational、navigational 和 transactional。

### Q16. Informational query 是什么？

A：用户想学习或了解某事，例如 Information Retrieval。

### Q17. Navigational query 是什么？

A：用户想去某个特定页面或网站，例如 United Airlines。

### Q18. Transactional query 是什么？

A：用户想完成某个 web-mediated action，如查天气、下载图片、购物。

### Q19. Exploratory search 是什么？

A：用户想探索“有什么”，需求开放，边界不明确。

### Q20. 为什么 user intent 对 ranking 重要？

A：不同 intent 需要不同结果类型、排序特征和界面呈现。

## SEO

### Q21. SEO 是什么？

A：Search Engine Optimization，通过调整网页使其在 algorithmic search results 中对特定 keywords 排名更高。

### Q22. SEO 和 paid search 的区别是什么？

A：SEO 试图提高自然排名；paid search 通过付费获得广告位置。

### Q23. SEO 是否一定不合法或有害？

A：不是。有些 SEO 合法合理，有些则 shady 或 spammy。

### Q24. 早期 SEO 如何利用 TF-IDF ranking？

A：大量重复目标关键词，如 maui resort，多次堆砌以提高词频。

### Q25. Misleading meta-tags 是什么？

A：在网页 metadata 中加入与内容不符但能吸引搜索流量的关键词。

### Q26. 隐藏关键词堆砌是什么？

A：把重复关键词设成与背景同色，让 crawlers 看到但用户看不到。

### Q27. 为什么 pure word density 不能被信任？

A：它很容易被 SEO 操纵，不一定反映真实内容质量或 relevance。

### Q28. Cloaking 是什么？

A：对 search engine spider 提供不同于普通用户看到的 fake content，是 black-hat spam 技术。

## Duplicate Detection

### Q29. 为什么 web duplicate detection 重要？

A：web 充满 duplicate 和 near-duplicate content，若不去重会浪费 index、影响 ranking 和用户体验。

### Q30. Strict duplicate detection 是什么？

A：exact match 检测，可用 fingerprints 实现。

### Q31. Near-duplicate detection 是什么？

A：检测内容高度相似但不完全相同的 documents，例如只有 last modified date 不同。

### Q32. Near-duplicate detection 常用什么标准？

A：similarity threshold，例如 similarity > 80%。

### Q33. Near-duplicate relation 是否传递？

A：不一定。A≈B 且 B≈C 不保证 A≈C。

### Q34. Shingles 是什么？

A：document 的 word n-grams，用作集合相似度特征。

### Q35. Shingle set similarity 常用什么？

A：Jaccard similarity，即 intersection size / union size。

### Q36. 为什么不能对所有 document pairs 精确算 shingle intersection？

A：web-scale pairwise comparison 计算量不可承受。

### Q37. MinHash/sketch 的核心思想是什么？

A：为每个 document 取短 sketch，用 sketch 近似估计 shingle sets 的 Jaccard similarity。

### Q38. MinHash 在 duplicate detection 中解决什么？

A：用小规模 signature 近似大集合相似度，降低 pairwise comparison 成本。

## Web Crawling

### Q39. Web crawling 的基本流程是什么？

A：从 seed URLs 开始，fetch and parse，extract URLs，放入 queue/frontier，重复抓取。

### Q40. URL frontier 是什么？

A：等待抓取的 URLs 队列/集合，需管理优先级、去重和 politeness。

### Q41. Dark Web 在 crawling 图中表示什么？

A：crawler 不能直接从已知 links 到达或未被发现的 web 部分。

### Q42. Crawler must be polite 是什么意思？

A：尊重 robots.txt，并避免过于频繁访问同一 server。

### Q43. Crawler must be robust 是什么意思？

A：能抵抗 spider traps、malicious servers、spam/link farms 等异常行为。

### Q44. Crawler should be scalable 是什么意思？

A：能通过增加机器提高 crawl rate，并支持 distributed operation。

### Q45. Crawler freshness 是什么？

A：持续重新抓取已抓页面的新版本，保持 index 更新。

### Q46. Basic crawler architecture 包括哪些组件？

A：DNS、fetch、parse、content seen check、URL filter、duplicate URL elimination、robots filters、URL frontier、fingerprints 等。

### Q47. Crawling processing steps 是什么？

A：从 frontier 取 URL，fetch document，parse/extract links，检查 content duplicate，加入 index，再过滤和去重 extracted URLs。

### Q48. Explicit politeness 是什么？

A：遵守 webmaster 在 robots.txt 中指定的 crawl rules。

### Q49. Implicit politeness 是什么？

A：即使无明确规则，也不要过于频繁请求同一 host。

### Q50. URL frontier 的两个主要考虑是什么？

A：politeness 和 priority/freshness。

### Q51. 为什么 priority 和 politeness 可能冲突？

A：高优先级 URLs 可能集中在同一 host，简单优先队列会造成对该 host 的访问突发。

### Q52. 常见 politeness heuristic 是什么？

A：对同一 host 的连续请求之间插入时间间隔，通常大于最近一次 fetch 所需时间。

## 考前作答模板

### Q53. 如果考“web crawler”，应该怎么答？

A：从 seed URLs 开始，fetch/parse/extract links，维护 URL frontier，做 URL/content dedup、robots filtering、politeness 和 freshness management。

### Q54. 如果考“SEO spam”，应该怎么答？

A：解释 SEO 是提升自然排名；早期 keyword stuffing、hidden text、misleading meta-tags、cloaking 等利用 ranking signals，说明 pure word density 不可靠。

### Q55. 如果考“near duplicate detection”，应该怎么答？

A：用 shingles 表示 documents，Jaccard 衡量相似度，web-scale 下用 MinHash/sketch 近似，避免全量 pairwise exact comparison。

