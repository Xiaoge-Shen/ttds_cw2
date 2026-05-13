# TTDS: Lab 6

Based on lectures 12 and 13 ("Comparing Corpora").

## SETUP

- Download and unzip the [two mystery corpora](https://uoe-my.sharepoint.com/:u:/g/personal/bross3_ed_ac_uk/Ea-j5sK1TuxLtLa7UCzDPKcBYG7IZVY6GVgcPTZhkMIxzg?e=6VhWYw).
  - Each corpus contains posts from a single subcommunity on Reddit.com ("subreddit").
  - The boundaries between posts are not obvious in the files. You can split the contents on empty lines, treating each paragraph as a separate document.
- Apply preprocessing: tokenization, stopping, case folding, stemming.
  - You can reuse your lab 1 code.
  - You should also remove any strings that appear less than 10 times across both corpora.

## WORD-LEVEL COMPARISONS

- Now you should find the most distinctive words for each corpus.
- Write code in any programming language to compute **Mutual Information (MI)** and **Chi-squared (X**2**)** values for all words in each corpus.
  - You can ignore words that appear less than 10 times overall.
  - Hint: if you compute N00, N01, N11, and N10 (as described in the lecture) once for each word-corpus pair, you can use these numbers to quickly get both **MI** and **X2.**
- Produce a sorted list of words by **MI** and **X2** for each corpus.
- Examine the top ten **MI** words and top ten **X2** words for each corpus.
- Discuss with a friend:
  - What are the differences you noticed between the two methods?
  - What did you learn about each corpus by analyzing these lists?

## TOPIC-LEVEL COMPARISONS

- Install the [gensim](https://radimrehurek.com/gensim/) (follow this link) python library.
  - see section titled “Quick Install”.
  - Note: you can pass the “--user” flag to pip to install a package to your home directory instead of doing a system-wide install.
- Treating each blank line in each corpus file as the start of a new *document*,
  - train an LDA model with 10 topics on these *documents* as described [here](https://radimrehurek.com/gensim/models/ldamodel.html).
    - See “Usage Examples”: follow the steps in the first block of code.
  - common_texts should be a list of your *documents* in this list-of-lists format:
    - \[ \[‘this’, ‘is’, ‘a’, ‘sample’, ‘document’\] , \[‘this’, ‘is’, ‘another’\] \]
      - Use your preprocessed, tokenized text with stopwords removed.
    - Using the get_document_topics() function of the lda model object, get the topic scores for each *document*, then compute **overall topic probabilities** for each corpus by averaging the topic probabilities for all documents belonging to each corpus.
    - Determine the top 3 topics for each corpus based on **overall probability**.
      - View the top 10 words for these topics using the [print_topic()](https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.print_topic)
    - If you have extra time, try changing the number of topics and running lda again to see what happens, or try creating a visualization of your topics using [pyLDAvis](https://github.com/bmabey/pyLDAvis)!
