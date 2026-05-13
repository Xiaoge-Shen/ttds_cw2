# TTDS Coursework 2: IR Evaluation, Text Analysis, and Classification
# Student ID: s2414220
# This code implements all three parts of the coursework assignment

# Standard libraries for data manipulation and numerical computation
import pandas as pd
import numpy as np
import math
from scipy import stats
import csv
from collections import defaultdict, Counter
import re

# Natural language processing libraries
import nltk
from nltk.corpus import stopwords

# Machine learning and feature extraction libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

# Uncomment below if NLTK data needs to be downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# ==========================================
# Part 1: IR Evaluation
# ==========================================

class IREvaluator:
    def __init__(self, results_path, qrels_path):
        self.results = pd.read_csv(results_path)
        self.qrels = pd.read_csv(qrels_path)
        # Convert qrels to dictionary format: {query_id: {doc_id: relevance_score}}
        self.qrels_dict = self._process_qrels()

    def _process_qrels(self):
        """Helper function: Process qrels into dictionary structure"""
        qrels_dict = defaultdict(dict)
        for idx, row in self.qrels.iterrows():
            query_id = row['query_id']
            doc_id = row['doc_id']
            relevance = row['relevance']
            qrels_dict[query_id][doc_id] = relevance
        return qrels_dict

    def _precision_at_k(self, retrieved_docs, query_id, k):
        """Calculate Precision at K"""
        if query_id not in self.qrels_dict:
            return 0
        top_k_docs = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k_docs if doc in self.qrels_dict[query_id])
        return relevant_count / k if k > 0 else 0.0

    def _recall_at_k(self, retrieved_docs, query_id, k):
        """Calculate Recall at K"""
        if query_id not in self.qrels_dict:
            return 0
        relevant_docs = set(self.qrels_dict[query_id].keys())
        total_relevant = len(relevant_docs)
        
        if total_relevant == 0:
            return 0.0
            
        top_k_docs = retrieved_docs[:k]
        retrieved_relevant = sum(1 for doc in top_k_docs 
                                if doc in relevant_docs)
        
        return retrieved_relevant / total_relevant

    def _r_precision(self, retrieved_docs, query_id):
        """Calculate R-Precision"""
        if query_id not in self.qrels_dict:
            return 0.0
        r = len(self.qrels_dict[query_id])
        return self._precision_at_k(retrieved_docs, query_id, r)

    def _average_precision(self, retrieved_docs, query_id):
        """Calculate Average Precision (AP)"""
        if query_id not in self.qrels_dict:
            return 0.0
        r_docs = self.qrels_dict[query_id]
        r = len(r_docs)
        precision_sum = 0.0
        relevant_count = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in r_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        return precision_sum / r

    def _ndcg_at_k(self, retrieved_docs, query_id, k):
        """Calculate normalized Discounted Cumulative Gain at K (nDCG@K)"""
        if query_id not in self.qrels_dict:
            return 0.0
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], 1):
            rel = self.qrels_dict[query_id].get(doc, 0)
            if i == 1:
                dcg += rel
            else:
                dcg += rel / math.log2(i)
        idcg = 0.0
        i_rels = sorted(self.qrels_dict[query_id].values(), reverse=True)[:k]
        for i, rel in enumerate(i_rels, 1):
            if i == 1:
                idcg += rel
            else:
                idcg += rel / math.log2(i)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_all(self, output_file="ir_eval.csv"):
        """Main evaluation loop: Evaluate all systems and generate CSV output"""
        systems = self.results['system_number'].unique()
        queries = self.results['query_number'].unique()
        
        output_data = []

        for sys_id in systems:
            sys_rows = []  # Store results for all queries of this system to compute mean
            
            for q_id in queries:
                # Get retrieval results for current system and query, sorted by rank
                current_results = self.results[(self.results['system_number'] == sys_id) & 
                                               (self.results['query_number'] == q_id)]
                sorted_docs = current_results.sort_values('rank_of_doc')['doc_number'].tolist()
                
                # Calculate all metrics
                p10 = self._precision_at_k(sorted_docs, q_id, 10)
                r50 = self._recall_at_k(sorted_docs, q_id, 50)
                r_prec = self._r_precision(sorted_docs, q_id)
                ap = self._average_precision(sorted_docs, q_id)
                ndcg10 = self._ndcg_at_k(sorted_docs, q_id, 10)
                ndcg20 = self._ndcg_at_k(sorted_docs, q_id, 20)
                
                row = [sys_id, q_id, p10, r50, r_prec, ap, ndcg10, ndcg20]
                sys_rows.append(row)
                output_data.append(row)
            
            mean_metrics = [np.mean([row[i] for row in sys_rows]) for i in range(2, len(row))]
            output_data.append([sys_id, "mean"] + mean_metrics)

        # Write to CSV with 3 decimal places
        headers = ["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"]
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in output_data:
                formatted_row = [row[0], row[1]] + [round(val, 3) for val in row[2:]]
                writer.writerow(formatted_row)
        print(f"IR Evaluation results saved to {output_file}")

    def perform_significance_test(self, ir_eval_file="ir_eval.csv"):
        """Perform statistical significance testing using t-test on evaluation results"""
        # Read evaluation results from CSV
        eval_results = pd.read_csv(ir_eval_file)
        
        # Filter out mean rows, keep only individual query results
        eval_results = eval_results[eval_results['query_number'] != 'mean']
        
        metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
        systems = sorted(eval_results['system_number'].unique())
        
        # Collect metric scores for each system
        system_scores = defaultdict(lambda: defaultdict(list))
        
        for sys_id in systems:
            sys_data = eval_results[eval_results['system_number'] == sys_id]
            for metric in metrics:
                system_scores[sys_id][metric] = sys_data[metric].tolist()
        
        # Find best and second-best systems for each metric
        print("\n=== Significance Test Results ===\n")
        for metric in metrics:
            # Calculate mean score for each system
            avg_scores = {sys_id: np.mean(system_scores[sys_id][metric]) 
                        for sys_id in systems}
            # Sort by score
            ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            best_sys, best_score = ranked[0]
            second_sys, second_score = ranked[1]
            
            # Perform two-tailed t-test
            t_stat, p_value = stats.ttest_ind(
                system_scores[best_sys][metric],
                system_scores[second_sys][metric]
            )
            
            significant = "YES" if p_value < 0.05 else "NO"
            
            print(f"{metric}:")
            print(f"  Best: System {best_sys} (mean={best_score:.3f})")
            print(f"  Second: System {second_sys} (mean={second_score:.3f})")
            print(f"  T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
            print(f"  Significantly better? {significant}")
            print()

# ==========================================
# Part 2: Text Analysis
# ==========================================

class TextAnalyzer:
    def __init__(self, data_path):
        """
        Initialize TextAnalyzer with corpus data.
        Args:
            data_path: Path to TSV file containing all three corpora
        """
        self.data_path = data_path
        self.raw_corpora = {'Quran': [], 'OT': [], 'NT': []}
        self.processed_corpora = {'Quran': [], 'OT': [], 'NT': []}
        self.vocab = set()  
        self._load_data()
        
    def _load_data(self):
        """Load TSV file and organize by corpus"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    corpus_name, text = parts
                    if corpus_name in self.raw_corpora:
                        self.raw_corpora[corpus_name].append(text)
        
        print(f"Loaded data:")
        for name, docs in self.raw_corpora.items():
            print(f"  {name}: {len(docs)} documents")   

    def preprocess(self):
        """Preprocess corpora: tokenization, lowercasing, stopword removal"""
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        for corpus_name, documents in self.raw_corpora.items():
            processed_docs = []
            for doc in documents:
                # Convert to lowercase
                doc = doc.lower()
                # Tokenize (extract alphabetic tokens only)
                tokens = re.findall(r'\b[a-z]+\b', doc)
                # Remove stopwords and short tokens (length <= 2)
                tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
                processed_docs.append(tokens)
                self.vocab.update(tokens)
            
            self.processed_corpora[corpus_name] = processed_docs
        
        print(f"\nPreprocessing complete:")
        print(f"  Total vocabulary size: {len(self.vocab)}")
        for name, docs in self.processed_corpora.items():
            total_tokens = sum(len(doc) for doc in docs)
            print(f"  {name}: {total_tokens} tokens")

    def compute_feature_selection(self):
        """Compute Mutual Information and Chi-Square scores for feature selection"""
        
        results = {'MI': {}, 'Chi2': {}}
        
        for target_corpus in ['Quran', 'OT', 'NT']:
            print(f"\n{'='*50}")
            print(f"Computing feature selection for: {target_corpus}")
            print(f"{'='*50}")
            
            mi_scores = {}
            chi2_scores = {}
            
            # Calculate scores for each word
            for word in self.vocab:
                # Build contingency table
                # N11: documents in target_corpus containing the word
                # N10: documents in target_corpus NOT containing the word
                # N01: documents in other_corpora containing the word
                # N00: documents in other_corpora NOT containing the word
                
                target_docs = self.processed_corpora[target_corpus]
                other_docs = []
                for corpus_name, docs in self.processed_corpora.items():
                    if corpus_name != target_corpus:
                        other_docs.extend(docs)
                
                # Calculate contingency table values
                N11 = sum(1 for doc in target_docs if word in doc)
                N10 = len(target_docs) - N11
                N01 = sum(1 for doc in other_docs if word in doc)
                N00 = len(other_docs) - N01
                
                N = N11 + N10 + N01 + N00  # Total number of documents
                
                # Skip words that never appear
                if N11 == 0 and N01 == 0:
                    continue
                
                # Calculate Mutual Information
                # Formula: MI = log2(P(w,c) / (P(w) * P(c)))
                # Alternative: MI = sum over all cells: (Nij/N) * log2((N*Nij)/(Ni*Nj))
                
                # Simplified MI calculation for single class
                if N11 > 0:
                    # P(word, target_class)
                    p_wc = N11 / N
                    # P(word)
                    p_w = (N11 + N01) / N
                    # P(target_class)
                    p_c = (N11 + N10) / N
                    
                    if p_w > 0 and p_c > 0:
                        mi = math.log2(p_wc / (p_w * p_c))
                        mi_scores[word] = mi
                
                # Calculate Chi-Square
                # Formula: χ² = N * (N11*N00 - N10*N01)² / ((N11+N10)(N11+N01)(N10+N00)(N01+N00))
                association = N11 * N00 - N10 * N01
                numerator = N * association ** 2
                denominator = (N11 + N10) * (N11 + N01) * (N10 + N00) * (N01 + N00)

                # Only keep positive associations (words more frequent in target corpus)
                if denominator > 0 and association > 0:
                    chi2 = numerator / denominator
                    chi2_scores[word] = chi2
            
            # Sort and get top 10 words
            top_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            top_chi2 = sorted(chi2_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            results['MI'][target_corpus] = top_mi
            results['Chi2'][target_corpus] = top_chi2
            
            # 打印结果
            print(f"\nTop 10 words by Mutual Information:")
            for i, (word, score) in enumerate(top_mi, 1):
                print(f"  {i:2d}. {word:15s} {score:.4f}")
            
            print(f"\nTop 10 words by Chi-Square:")
            for i, (word, score) in enumerate(top_chi2, 1):
                print(f"  {i:2d}. {word:15s} {score:.4f}")
        
        return results

    def run_lda_analysis(self, num_topics=20):
        """Run Latent Dirichlet Allocation (LDA) topic modeling"""
        print(f"\n{'='*50}")
        print(f"Running LDA with {num_topics} topics")
        print(f"{'='*50}")
        
        # 1. Prepare data: combine all documents while tracking their corpus
        all_docs = []
        doc_labels = []  # Track which corpus each document belongs to
        
        for corpus_name in ['Quran', 'OT', 'NT']:
            for doc in self.processed_corpora[corpus_name]:
                all_docs.append(' '.join(doc))  # Convert token list back to string
                doc_labels.append(corpus_name)
        
        print(f"Total documents for LDA: {len(all_docs)}")
        
        # 2. Build document-term matrix
        vectorizer = CountVectorizer(max_features=5000, min_df=5)
        doc_term_matrix = vectorizer.fit_transform(all_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"Vocabulary size: {len(feature_names)}")
        print(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        # 3. Train LDA model
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        
        print("Training LDA model...")
        doc_topic_dist = lda_model.fit_transform(doc_term_matrix)
        print("LDA training complete!")
        
        # 4. Calculate average topic distribution for each corpus
        corpus_topic_scores = defaultdict(lambda: np.zeros(num_topics))
        corpus_doc_counts = defaultdict(int)
        
        for doc_idx, corpus_name in enumerate(doc_labels):
            corpus_topic_scores[corpus_name] += doc_topic_dist[doc_idx]
            corpus_doc_counts[corpus_name] += 1
        
        # Compute averages
        for corpus_name in corpus_topic_scores:
            corpus_topic_scores[corpus_name] /= corpus_doc_counts[corpus_name]
        
        # 5. Find most prominent topic for each corpus
        print(f"\n{'='*50}")
        print("Most Prominent Topics for Each Corpus")
        print(f"{'='*50}")
        
        results = {}
        
        for corpus_name in ['Quran', 'OT', 'NT']:
            # Find topic with highest score for this corpus
            top_topic_idx = np.argmax(corpus_topic_scores[corpus_name])
            top_topic_score = corpus_topic_scores[corpus_name][top_topic_idx]
            
            # Get top 10 words for this topic
            topic_words_dist = lda_model.components_[top_topic_idx]
            top_word_indices = topic_words_dist.argsort()[-10:][::-1]
            top_words = [(feature_names[i], topic_words_dist[i]) 
                        for i in top_word_indices]
            
            results[corpus_name] = {
                'topic_id': top_topic_idx,
                'topic_score': top_topic_score,
                'top_words': top_words
            }
            
            print(f"\n{corpus_name}:")
            print(f"  Most prominent topic: Topic {top_topic_idx}")
            print(f"  Average score: {top_topic_score:.4f}")
            print(f"  Top 10 words:")
            for i, (word, prob) in enumerate(top_words, 1):
                print(f"    {i:2d}. {word:15s} {prob:.4f}")
        
        # 6. Analyze cross-corpus topic distribution
        print(f"\n{'='*50}")
        print("Cross-Corpus Topic Analysis")
        print(f"{'='*50}")
        
        # Display topic distribution for each corpus
        print("\nTopic distribution for each corpus:")
        print(f"{'Topic':<10}", end='')
        for corpus_name in ['Quran', 'OT', 'NT']:
            print(f"{corpus_name:<12}", end='')
        print()
        print("-" * 40)
        
        for topic_idx in range(num_topics):
            print(f"Topic {topic_idx:<4}", end='')
            for corpus_name in ['Quran', 'OT', 'NT']:
                score = corpus_topic_scores[corpus_name][topic_idx]
                print(f"{score:<12.4f}", end='')
            print()
        
        return results, lda_model, vectorizer, corpus_topic_scores

# ==========================================
# Part 3: Text Classification
# ==========================================

class SentimentClassifier:
    def __init__(self, train_path, test_path=None):
        """
        Initialize sentiment classifier with training and optional test data.
        Args:
            train_path: Path to training data file
            test_path: Path to test data file (optional, can be loaded later)
        """
        # Load training data (TSV format with header: id, sentiment, tweet)
        self.train_data = pd.read_csv(train_path, sep='\t', header=0)
        # Data format: id, sentiment, tweet
        
        self.test_data = None
        if test_path:
            try:
                self.test_data = pd.read_csv(test_path, sep='\t', header=0)
            except:
                print("Test file not found, will load later")
        
        # Data splits
        self.X_train = None
        self.X_dev = None
        self.y_train = None
        self.y_dev = None
        self.X_test = None
        self.y_test = None
        
        # Feature vectorizers
        self.vectorizer = None
        self.vectorizer_improved = None
        
        # Classification models
        self.baseline_model = None
        self.improved_model = None
        
        # Results storage
        self.results = []
        
    def split_data(self, test_size=0.1, random_state=42):
        """Shuffle and split data into training and development sets"""
        from sklearn.model_selection import train_test_split
        
        # Extract features (X) from 'tweet' column and labels (y) from 'sentiment' column
        X = self.train_data['tweet'].values
        y = self.train_data['sentiment'].values
        
        # Check class distribution
        print(f"\nOriginal class distribution:")
        print(pd.Series(y).value_counts())
        
        # Use stratified split if all classes have sufficient samples
        class_counts = pd.Series(y).value_counts()
        use_stratify = all(count >= 2 for count in class_counts.values)
        
        if use_stratify:
            self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
        else:
            print("Warning: Some classes have too few samples. Stratification disabled.")
            self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"\nData split complete:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Development set: {len(self.X_dev)} samples")
        print(f"  Class distribution in train:")
        print(pd.Series(self.y_train).value_counts())
        
    def load_test_data(self, test_path):
        """Load test dataset from file"""
        self.test_data = pd.read_csv(test_path, sep='\t', header=0)
        # Test data format: id, sentiment, tweet
        self.X_test = self.test_data['tweet'].values
        self.y_test = self.test_data['sentiment'].values
        print(f"Test set loaded: {len(self.X_test)} samples")
        print(f"Test class distribution:")
        print(pd.Series(self.y_test).value_counts())
    
    def extract_features_baseline(self):
        """Extract baseline features using Bag-of-Words (BOW) representation"""
        print("\n" + "="*50)
        print("Extracting Baseline Features (BOW)")
        print("="*50)
        
        # Use CountVectorizer for BOW representation
        self.vectorizer = CountVectorizer(
            lowercase=True,
            max_features=10000,  # Limit vocabulary size
            min_df=2  # Minimum document frequency (at least 2 occurrences)
        )
        
        # Fit on training set and transform
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        # Transform development set using same vocabulary
        X_dev_vec = self.vectorizer.transform(self.X_dev)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Training matrix shape: {X_train_vec.shape}")
        print(f"Dev matrix shape: {X_dev_vec.shape}")
        
        return X_train_vec, X_dev_vec

    def train_baseline(self):
        """Train baseline SVM classifier with C=1000 as specified in assignment"""
        from sklearn.svm import LinearSVC
        
        print("\n" + "="*50)
        print("Training Baseline Model: Linear SVM (C=1000)")
        print("="*50)
        
        # Extract BOW features
        X_train_vec, X_dev_vec = self.extract_features_baseline()
        
        # Train Linear SVM with C=1000
        self.baseline_model = LinearSVC(C=1000, random_state=42, max_iter=1000)
        self.baseline_model.fit(X_train_vec, self.y_train)
        
        print("Baseline model training complete!")
        
        # Evaluate on training set
        print("\n--- Baseline Performance on Train Set ---")
        train_results = self.evaluate(self.baseline_model, X_train_vec, 
                                     self.y_train, split_name='train')
        self.results.append(['baseline', 'train'] + train_results)
        
        # Evaluate on development set
        print("\n--- Baseline Performance on Dev Set ---")
        dev_results = self.evaluate(self.baseline_model, X_dev_vec, 
                                    self.y_dev, split_name='dev')
        self.results.append(['baseline', 'dev'] + dev_results)
        
        return self.baseline_model
    
    def evaluate(self, model, X, y_true, split_name=''):
        """Calculate Precision, Recall, and F1-score (per-class and macro-averaged)"""
        from sklearn.metrics import precision_recall_fscore_support, classification_report
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=['positive', 'negative', 'neutral'], 
            average=None, zero_division=0
        )
        
        # Calculate macro-averaged metrics
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Print detailed classification report
        print(classification_report(y_true, y_pred, 
                                   labels=['positive', 'negative', 'neutral'],
                                   digits=3))
        
        # Return format: [p-pos, r-pos, f-pos, p-neg, r-neg, f-neg, 
        #                 p-neu, r-neu, f-neu, p-macro, r-macro, f-macro]
        results = [
            precision[0], recall[0], f1[0],  # positive class
            precision[1], recall[1], f1[1],  # negative class
            precision[2], recall[2], f1[2],  # neutral class
            p_macro, r_macro, f_macro         # macro-averaged
        ]
        
        return results
    
    def analyze_errors(self, num_examples=3):
        """Analyze misclassified examples from development set for error pattern identification"""
        print("\n" + "="*50)
        print("Error Analysis on Dev Set")
        print("="*50)
        
        # Make predictions on development set
        X_dev_vec = self.vectorizer.transform(self.X_dev)
        y_pred = self.baseline_model.predict(X_dev_vec)
        
        # Identify misclassified samples
        errors = []
        for i, (true, pred) in enumerate(zip(self.y_dev, y_pred)):
            if true != pred:
                errors.append({
                    'index': i,
                    'text': self.X_dev[i],
                    'true': true,
                    'pred': pred
                })
        
        print(f"\nTotal errors: {len(errors)} / {len(self.y_dev)}")
        print(f"Accuracy: {1 - len(errors)/len(self.y_dev):.3f}")
        
        # Randomly sample examples for detailed analysis
        import random
        random.seed(42)
        sample_errors = random.sample(errors, min(num_examples, len(errors)))
        
        print(f"\n{num_examples} Example Errors:")
        for i, err in enumerate(sample_errors, 1):
            print(f"\n{i}. Text: {err['text'][:100]}...")
            print(f"   True: {err['true']} | Predicted: {err['pred']}")
        
        return sample_errors

    def train_improved_model(self):
        """Train improved classifier with enhanced features and tuned hyperparameters"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        
        print("\n" + "="*50)
        print("Training Improved Model")
        print("="*50)
        
        # Improvement strategies:
        # 1. TF-IDF weighting instead of raw BOW counts
        # 2. Bigram features (unigrams + bigrams) to capture multi-word expressions
        # 3. Reduced regularization parameter (C=500) to mitigate overfitting
        # 4. Sublinear TF scaling for better feature normalization
        
        print("Improvements:")
        print("  1. TF-IDF features (instead of BOW)")
        print("  2. Bigrams (unigrams + bigrams)")
        print("  3. Fine-tuned C parameter (C=500)")
        print("  4. Sublinear TF scaling")
        
        self.vectorizer_improved = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            max_features=20000,
            min_df=2,
            sublinear_tf=True,  # Use log(1+tf) scaling
            max_df=0.95  # Ignore terms appearing in >95% of documents
        )
        
        # Extract improved features
        X_train_vec = self.vectorizer_improved.fit_transform(self.X_train)
        X_dev_vec = self.vectorizer_improved.transform(self.X_dev)
        
        print(f"Improved vocabulary size: {len(self.vectorizer_improved.vocabulary_)}")
        print(f"Training matrix shape: {X_train_vec.shape}")
        
        # Train improved SVM with reduced C
        self.improved_model = LinearSVC(C=500, random_state=42, max_iter=2000)
        self.improved_model.fit(X_train_vec, self.y_train)
        
        print("Improved model training complete!")
        
        # Evaluate on training set
        print("\n--- Improved Performance on Train Set ---")
        train_results = self.evaluate(self.improved_model, X_train_vec, 
                                     self.y_train, split_name='train')
        self.results.append(['improved', 'train'] + train_results)
        
        # Evaluate on development set
        print("\n--- Improved Performance on Dev Set ---")
        dev_results = self.evaluate(self.improved_model, X_dev_vec, 
                                    self.y_dev, split_name='dev')
        self.results.append(['improved', 'dev'] + dev_results)
        
        # Calculate improvement over baseline
        baseline_f_macro = self.results[1][-1]  # Baseline dev macro-f1
        improved_f_macro = dev_results[-1]
        improvement = improved_f_macro - baseline_f_macro
        
        print(f"\n>>> Improvement on Dev Set:")
        print(f"    Baseline Macro-F1: {baseline_f_macro:.3f}")
        print(f"    Improved Macro-F1: {improved_f_macro:.3f}")
        print(f"    Gain: +{improvement:.3f}")
        
        return self.improved_model
    
    def evaluate_on_test(self):
        """Evaluate both baseline and improved models on test set"""
        if self.X_test is None:
            print("Test set not loaded!")
            return
        
        print("\n" + "="*50)
        print("Evaluating on Test Set")
        print("="*50)
        
        # Evaluate baseline on test set
        X_test_baseline = self.vectorizer.transform(self.X_test)
        print("\n--- Baseline Performance on Test Set ---")
        test_results_baseline = self.evaluate(self.baseline_model, 
                                             X_test_baseline, 
                                             self.y_test, split_name='test')
        self.results.append(['baseline', 'test'] + test_results_baseline)
        
        # Evaluate improved model on test set
        X_test_improved = self.vectorizer_improved.transform(self.X_test)
        print("\n--- Improved Performance on Test Set ---")
        test_results_improved = self.evaluate(self.improved_model, 
                                             X_test_improved, 
                                             self.y_test, split_name='test')
        self.results.append(['improved', 'test'] + test_results_improved)
        
        # Calculate and display improvement
        baseline_f_macro = test_results_baseline[-1]
        improved_f_macro = test_results_improved[-1]
        improvement = improved_f_macro - baseline_f_macro
        
        print(f"\n>>> Improvement on Test Set:")
        print(f"    Baseline Macro-F1: {baseline_f_macro:.3f}")
        print(f"    Improved Macro-F1: {improved_f_macro:.3f}")
        print(f"    Gain: +{improvement:.3f}")

    def generate_output_csv(self, output_file="classification.csv"):
        """Generate final classification results CSV file for submission"""
        headers = ["system", "split", "p-pos", "r-pos", "f-pos", 
                   "p-neg", "r-neg", "f-neg", "p-neu", "r-neu", "f-neu", 
                   "p-macro", "r-macro", "f-macro"]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in self.results:
                # Round all metrics to 3 decimal places as required
                formatted_row = row[:2] + [round(x, 3) for x in row[2:]]
                writer.writerow(formatted_row)
        
        print(f"\nClassification results saved to {output_file}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # ==========================================
    # TTDS Coursework 2 - Main Execution
    # ==========================================
    
    # Part 1: IR System Evaluation
    # Evaluates 6 IR systems on 10 queries using P@K, R@K, R-Precision, AP, and nDCG
    # Generates ir_eval.csv and performs significance testing
    ir = IREvaluator("ttdssystemresults.csv", "qrels.csv")
    ir.evaluate_all()
    ir.perform_significance_test() 

    # Part 2: Text Analysis
    # Computes Mutual Information, Chi-Square, and performs LDA topic modeling
    # on three religious text corpora (Quran, Old Testament, New Testament)
    analyzer = TextAnalyzer("bible_and_quran.tsv")
    analyzer.preprocess()
    analyzer.compute_feature_selection()
    analyzer.run_lda_analysis()

    # Part 3: Sentiment Classification
    # Three-way sentiment classification (positive, negative, neutral) on tweets
    # Trains baseline (BOW+SVM) and improved (TF-IDF+Bigrams+SVM) models
    # Generates classification.csv
    clf = SentimentClassifier("train.txt")  
    clf.split_data(test_size=0.1, random_state=42)
    clf.train_baseline()
    error_examples = clf.analyze_errors(num_examples=3)
    clf.train_improved_model()
    try:
        clf.load_test_data("ttds_2025_cw2_test.txt") 
        clf.evaluate_on_test()
    except:
        print("\nTest set not available yet")
    clf.generate_output_csv("classification.csv")