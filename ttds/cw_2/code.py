import pandas as pd
import numpy as np
import math
from scipy import stats
import csv
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re

# nltk.download('punkt')
# nltk.download('stopwords')

# ==========================================
# Part 1: IR Evaluation
# ==========================================

class IREvaluator:
    def __init__(self, results_path, qrels_path):
        self.results = pd.read_csv(results_path)
        self.qrels = pd.read_csv(qrels_path)
        # 提示：将qrels转换为更易查询的字典格式，例如 {query_id: {doc_id: relevance_score}}
        self.qrels_dict = self._process_qrels()

    def _process_qrels(self):
        """辅助函数：处理qrels数据结构"""
        qrels_dict = defaultdict(dict)
        for idx, row in self.qrels.iterrows():
            query_id = row['query_id']
            doc_id = row['doc_id']
            relevance = row['relevance']
            qrels_dict[query_id][doc_id] = relevance
        return qrels_dict

    def _precision_at_k(self, retrieved_docs, query_id, k):
        """计算 P@K"""
        if query_id not in self.qrels_dict:
            return 0
        top_k_docs = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k_docs if doc in self.qrels_dict[query_id])
        return relevant_count / k if k > 0 else 0.0

    def _recall_at_k(self, retrieved_docs, query_id, k):
        """计算 R@K"""
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
        """计算 R-Precision"""
        if query_id not in self.qrels_dict:
            return 0.0
        r = len(self.qrels_dict[query_id])
        return self._precision_at_k(retrieved_docs, query_id, r)

    def _average_precision(self, retrieved_docs, query_id):
        """计算 AP"""
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
        """计算 nDCG@K"""
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
        """主循环：评估所有系统并生成 CSV"""
        systems = self.results['system_number'].unique()
        queries = self.results['query_number'].unique()
        
        output_data = []

        for sys_id in systems:
            sys_rows = [] # 用于存储该系统所有查询的结果，以便计算 mean
            
            for q_id in queries:
                # 获取当前系统、当前查询的检索结果，按 rank 排序
                current_results = self.results[(self.results['system_number'] == sys_id) & 
                                               (self.results['query_number'] == q_id)]
                sorted_docs = current_results.sort_values('rank_of_doc')['doc_number'].tolist()
                
                # 计算各项指标
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

        # 写入 CSV，注意保留3位小数
        headers = ["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"]
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in output_data:
                formatted_row = [row[0], row[1]] + [round(val, 3) for val in row[2:]]
                writer.writerow(formatted_row)
        print(f"IR Evaluation results saved to {output_file}")

    def perform_significance_test(self, ir_eval_file="ir_eval.csv"):
        """从已生成的ir_eval.csv直接读取结果进行统计检验"""
        # 读取评估结果
        eval_results = pd.read_csv(ir_eval_file)
        
        # 过滤掉mean行，只保留具体query的结果
        eval_results = eval_results[eval_results['query_number'] != 'mean']
        
        metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
        systems = sorted(eval_results['system_number'].unique())
        
        # 为每个系统收集各指标的分数
        system_scores = defaultdict(lambda: defaultdict(list))
        
        for sys_id in systems:
            sys_data = eval_results[eval_results['system_number'] == sys_id]
            for metric in metrics:
                system_scores[sys_id][metric] = sys_data[metric].tolist()
        
        # 对每个指标找出最好和第二好的系统
        print("\n=== Significance Test Results ===\n")
        for metric in metrics:
            # 计算每个系统的平均分
            avg_scores = {sys_id: np.mean(system_scores[sys_id][metric]) 
                        for sys_id in systems}
            # 排序
            ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            best_sys, best_score = ranked[0]
            second_sys, second_score = ranked[1]
            
            # T-test (2-tailed)
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
        Args:
            data_path: TSV文件路径，包含所有三个语料库
        """
        self.data_path = data_path
        self.raw_corpora = {'Quran': [], 'OT': [], 'NT': []}
        self.processed_corpora = {'Quran': [], 'OT': [], 'NT': []}
        self.vocab = set()  # 所有词汇
        self._load_data()
        
    def _load_data(self):
        """读取TSV文件，按语料库分组"""
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
        """预处理：分词、小写化、去停用词"""
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        for corpus_name, documents in self.raw_corpora.items():
            processed_docs = []
            for doc in documents:
                # 转小写
                doc = doc.lower()
                # 分词（保留字母）
                tokens = re.findall(r'\b[a-z]+\b', doc)
                # 去停用词，去短词
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
        """计算 MI 和 Chi-square"""
        
        results = {'MI': {}, 'Chi2': {}}
        
        for target_corpus in ['Quran', 'OT', 'NT']:
            print(f"\n{'='*50}")
            print(f"Computing feature selection for: {target_corpus}")
            print(f"{'='*50}")
            
            mi_scores = {}
            chi2_scores = {}
            
            # 为每个词计算分数
            for word in self.vocab:
                # 构建列联表 (Contingency Table)
                # N11: word在target_corpus中出现的文档数
                # N10: word在target_corpus中不出现的文档数
                # N01: word在other_corpora中出现的文档数
                # N00: word在other_corpora中不出现的文档数
                
                target_docs = self.processed_corpora[target_corpus]
                other_docs = []
                for corpus_name, docs in self.processed_corpora.items():
                    if corpus_name != target_corpus:
                        other_docs.extend(docs)
                
                # 计算N11, N10, N01, N00
                N11 = sum(1 for doc in target_docs if word in doc)
                N10 = len(target_docs) - N11
                N01 = sum(1 for doc in other_docs if word in doc)
                N00 = len(other_docs) - N01
                
                N = N11 + N10 + N01 + N00  # 总文档数
                
                # 跳过从未出现的词
                if N11 == 0 and N01 == 0:
                    continue
                
                # 计算 Mutual Information
                # MI = log2(P(w,c) / (P(w) * P(c)))
                # 或者使用公式：MI = sum over all cells of: (Nij/N) * log2((N*Nij)/(Ni*Nj))
                
                # 简化的MI计算（针对单个类别）
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
                
                # 计算 Chi-Square
                # χ² = N * (N11*N00 - N10*N01)² / ((N11+N10)(N11+N01)(N10+N00)(N01+N00))
                association = N11 * N00 - N10 * N01
                numerator = N * association ** 2
                denominator = (N11 + N10) * (N11 + N01) * (N10 + N00) * (N01 + N00)

                if denominator > 0 and association > 0:
                    chi2 = numerator / denominator
                    chi2_scores[word] = chi2
            
            # 排序并获取top 10
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
        """运行 LDA 主题模型"""
        print(f"\n{'='*50}")
        print(f"Running LDA with {num_topics} topics")
        print(f"{'='*50}")
        
        # 1. 准备数据：合并所有文档，但记录每个文档属于哪个语料库
        all_docs = []
        doc_labels = []  # 记录每个文档的来源
        
        for corpus_name in ['Quran', 'OT', 'NT']:
            for doc in self.processed_corpora[corpus_name]:
                all_docs.append(' '.join(doc))  # 将token列表转回字符串
                doc_labels.append(corpus_name)
        
        print(f"Total documents for LDA: {len(all_docs)}")
        
        # 2. 构建文档-词矩阵
        vectorizer = CountVectorizer(max_features=5000, min_df=5)
        doc_term_matrix = vectorizer.fit_transform(all_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"Vocabulary size: {len(feature_names)}")
        print(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        # 3. 训练 LDA 模型
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        
        print("Training LDA model...")
        doc_topic_dist = lda_model.fit_transform(doc_term_matrix)
        print("LDA training complete!")
        
        # 4. 计算每个语料库的平均主题分布
        corpus_topic_scores = defaultdict(lambda: np.zeros(num_topics))
        corpus_doc_counts = defaultdict(int)
        
        for doc_idx, corpus_name in enumerate(doc_labels):
            corpus_topic_scores[corpus_name] += doc_topic_dist[doc_idx]
            corpus_doc_counts[corpus_name] += 1
        
        # 计算平均值
        for corpus_name in corpus_topic_scores:
            corpus_topic_scores[corpus_name] /= corpus_doc_counts[corpus_name]
        
        # 5. 为每个语料库找出最显著的主题
        print(f"\n{'='*50}")
        print("Most Prominent Topics for Each Corpus")
        print(f"{'='*50}")
        
        results = {}
        
        for corpus_name in ['Quran', 'OT', 'NT']:
            # 找到该语料库分数最高的主题
            top_topic_idx = np.argmax(corpus_topic_scores[corpus_name])
            top_topic_score = corpus_topic_scores[corpus_name][top_topic_idx]
            
            # 获取该主题的top 10词
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
        
        # 6. 分析跨语料库的共同主题
        print(f"\n{'='*50}")
        print("Cross-Corpus Topic Analysis")
        print(f"{'='*50}")
        
        # 显示每个语料库在所有主题上的分布
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
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path, sep='\t', names=['sentiment', 'text']) # 假设格式
        # 注意：test_path 只有在最后阶段才有
        self.test_data = None 

    def split_data(self):
        """打乱并切分 Train/Dev"""
        # TODO: 使用 train_test_split 切分 self.train_data
        pass

    def extract_features(self, method='bow'):
        """特征提取"""
        # TODO: 实例化 CountVectorizer 或 TfidfVectorizer
        # TODO: fit_transform 训练集，transform 开发集
        pass

    def train_baseline(self):
        """训练 Baseline SVM (C=1000)"""
        # TODO: 使用 sklearn.svm.SVC(kernel='linear', C=1000) 或 LinearSVC
        pass

    def train_improved_model(self):
        """训练改进模型"""
        # TODO: 你的改进策略 (例如: 不同的预处理, n-grams, 不同的C值, 不同的模型)
        pass

    def evaluate(self, model, X, y_true):
        """计算 P, R, F1 (Micro/Macro)"""
        # TODO: 使用 precision_recall_fscore_support
        # 返回格式需符合 classification.csv 的要求
        pass

    def generate_output_csv(self, output_file="classification.csv"):
        """生成最终的提交文件"""
        # TODO: 按照指定格式输出 Baseline 和 Improved 模型在 Train, Dev, Test 上的结果
        pass

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Run IR Evaluation
    ir = IREvaluator("ttdssystemresults.csv", "qrels.csv")
    ir.evaluate_all()
    ir.perform_significance_test() 

    # 2. Run Text Analysis
    analyzer = TextAnalyzer("bible_and_quran.tsv")
    analyzer.preprocess()
    analyzer.compute_feature_selection()
    analyzer.run_lda_analysis()

    # 3. Run Classification
    # clf = SentimentClassifier("train.txt", "test.txt")
    # clf.split_data()
    # clf.train_baseline()
    # ...
    pass