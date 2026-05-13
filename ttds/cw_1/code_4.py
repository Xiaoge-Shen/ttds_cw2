import re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
from math import log

def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f}
        return stopwords
    except FileNotFoundError:
        return set()  

def load_inverted_index(filepath):
    index = {}
    current_term = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.startswith('\t'):
                    # This is a term line: term:df
                    line = line.strip()
                    parts = line.split(':')
                    if len(parts) != 2:
                        print(f"Error on line {line_num}: Expected format 'term:df', got '{line}'")
                        continue
                    term, df = parts
                    current_term = term
                    index[current_term] = {}
                else:
                    # This is a document line: \tdoc_id: positions
                    line = line.strip()  # remove the leading tab
                    parts = line.split(':')
                    if len(parts) != 2:
                        print(f"Error on line {line_num}: Expected format 'doc_id:positions', got '{line}'")
                        continue
                    doc_id, positions = parts
                    doc_id = int(doc_id)
                    positions = [int(pos) for pos in positions.strip().split(',')]
                    index[current_term][doc_id] = positions
    except Exception as e:
        print(f"Error loading index: {e}")
        return None
    return index

def load_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f]
    return queries

def xml_parser(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    doc_data = {}
    for doc in root:
        docno = doc[0].text
        headline = doc.findtext('HEADLINE', '')
        text = doc.findtext('TEXT', '')
        doc_data[docno] = (headline + ' ' + text).strip()
    return doc_data

def preprocess_text(text, stopwords, stemmer):
    tokens = text.lower()
    pattern = re.compile(r'[a-z]+')
    tokens = pattern.findall(tokens)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def build_inverted_index(doc_data):
    inverted_index = {}
    for i, tokens in doc_data.items():
        for position, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if i not in inverted_index[token]:
                inverted_index[token][i] = []
            inverted_index[token][i].append(position)
    return inverted_index

def print_inverted_index(inverted_index):
    with open('index.txt', 'w', encoding='utf-8') as f:
        for token, positions in inverted_index.items():
            df = len(positions)
            print(f"{token}:{df}", file=f)
            for i, position in positions.items():
                positions_str = ','.join([str(pos) for pos in position])
                print(f"\t{i}: {positions_str}", file=f)

class BooleanSearchEngine:
    def __init__(self, inverted_index, stemmer, stopwords):
        self.inverted_index = inverted_index
        self.stemmer = stemmer
        self.stopwords = stopwords
    
    def process_query(self, query):
        """
        main query processing function - the unified entry point
        """
        # remove the query prefix (q1:, q2: etc.)
        if ':' in query:
            query = query.split(':', 1)[1].strip()
        
        # check if the query is a proximity query
        if re.match(r'#\d+\(', query):
            return self._handle_proximity_query(query)
        
        # parse the query
        operation, left, right = self.parse_query(query)
        
        # process the left operand
        left_docs = self.process_operand(left)
        
        # if there is no right operand, return the left operand
        if right is None:
            return list(left_docs)
        
        # process the right operand
        right_docs = self.process_operand(right)
        
        # execute the operation
        result = self.execute_operation(operation, left_docs, right_docs)
        
        return list(result)
    
    def parse_query(self, query):
        """
        parse the query, detect the operator and the operands
        return: (operation, left_part, right_part)
        """
        # detect the operator (by priority)
        if ' AND NOT ' in query:
            parts = query.split(' AND NOT ', 1)
            return 'AND NOT', parts[0].strip(), parts[1].strip()
        elif ' OR NOT ' in query:
            parts = query.split(' OR NOT ', 1)
            return 'OR NOT', parts[0].strip(), parts[1].strip()
        elif ' AND ' in query:
            parts = query.split(' AND ', 1)
            return 'AND', parts[0].strip(), parts[1].strip()
        elif ' OR ' in query:
            parts = query.split(' OR ', 1)
            return 'OR', parts[0].strip(), parts[1].strip()
        else:
            return None, query, None  # single query item
    
    def process_operand(self, operand):
        """
        process the operand: a term or a phrase
        """
        operand = operand.strip()
        
        # check if the operand is a phrase
        if operand.startswith('"') and operand.endswith('"'):
            return set(self._get_phrase_docs(operand))
        else:
            return set(self._get_term_docs(operand))
    
    def execute_operation(self, operation, left_docs, right_docs):
        if operation == 'AND':
            return left_docs & right_docs
        elif operation == 'OR':
            return left_docs | right_docs
        elif operation == 'AND NOT':
            return left_docs - right_docs
        elif operation == 'OR NOT':
            # A OR NOT B = A ∪ (U - B) where U is the universe
            all_docs = set()
            for term_docs in self.inverted_index.values():
                all_docs.update(term_docs.keys())
            return left_docs | (all_docs - right_docs)
        else:
            return set()
    
    def _get_term_docs(self, term):
        term = self._preprocess_term(term)
        if term in self.inverted_index:
            return list(self.inverted_index[term].keys())
        return []
    
    def _get_phrase_docs(self, phrase):
        # remove the quotes
        phrase = phrase.strip('"')
        terms = phrase.split()
        
        if len(terms) != 2:
            return []
        
        term1 = self._preprocess_term(terms[0])
        term2 = self._preprocess_term(terms[1])
        
        if term1 not in self.inverted_index or term2 not in self.inverted_index:
            return []
        
        result = []
        for doc_id in self.inverted_index[term1]:
            if doc_id in self.inverted_index[term2]:
                # check if the positions are consecutive
                for pos1 in self.inverted_index[term1][doc_id]:
                    if pos1 + 1 in self.inverted_index[term2][doc_id]:
                        result.append(doc_id)
                        break
        
        return result
    
    def _handle_proximity_query(self, query):
        match = re.match(r'#(\d+)\(([^,]+),\s*([^)]+)\)', query)
        if not match:
            return []
        
        max_distance = int(match.group(1))
        term1 = self._preprocess_term(match.group(2).strip())
        term2 = self._preprocess_term(match.group(3).strip())
        
        if term1 not in self.inverted_index or term2 not in self.inverted_index:
            return []
        
        result = []
        for doc_id in self.inverted_index[term1]:
            if doc_id in self.inverted_index[term2]:
                # check the distance between the positions
                for pos1 in self.inverted_index[term1][doc_id]:
                    for pos2 in self.inverted_index[term2][doc_id]:
                        if abs(pos1 - pos2) <= max_distance:
                            result.append(doc_id)
                            break
                    if doc_id in result:
                        break
        
        return result
    
    def _preprocess_term(self, term):
        """
        preprocess a single term: lowercase and stemming
        """
        term = term.lower().strip()
        return self.stemmer.stem(term)

class RankedRetrievalEngine:
    """
    Ranked retrieval engine using TF-IDF scoring
    """
    def __init__(self, inverted_index, stemmer, stopwords):
        self.inverted_index = inverted_index
        self.stemmer = stemmer
        self.stopwords = stopwords
        
        # Precomputed statistics
        self.N = 0                  # Total number of documents
        self.doc_lengths = {}       # Document lengths
        self.idf_scores = {}        # IDF scores for each term
        
        self._precompute_statistics()
    
    def _precompute_statistics(self):
        """
        Precompute document lengths and IDF scores
        """
        # Collect all document IDs
        all_docs = set()
        for postings in self.inverted_index.values():
            all_docs.update(postings.keys())
        
        self.N = len(all_docs)
        
        # Calculate document lengths
        for doc_id in all_docs:
            length = 0
            for term_postings in self.inverted_index.values():
                if doc_id in term_postings:
                    length += len(term_postings[doc_id])
            self.doc_lengths[doc_id] = length
        
        # Calculate IDF scores: log(N / df)
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf_scores[term] = log(self.N / df) if df > 0 else 0
    
    def _compute_tfidf(self, term, doc_id):
        """
        Compute TF-IDF score for a term in a document
        TF-IDF = (term_freq / doc_length) * log(N / df)
        """
        if term not in self.inverted_index:
            return 0.0
        if doc_id not in self.inverted_index[term]:
            return 0.0
        
        # TF: normalized term frequency
        tf = len(self.inverted_index[term][doc_id])
        if self.doc_lengths[doc_id] > 0:
            tf = tf / self.doc_lengths[doc_id]
        else:
            tf = 0
        
        # IDF
        idf = self.idf_scores.get(term, 0)
        
        return tf * idf
    
    def process_query(self, query_text):
        """
        Process a ranked query and return sorted documents with scores
        Returns: list of (doc_id, score) tuples sorted by score descending
        """
        # Preprocess query
        query_terms = self._preprocess_query(query_text)
        
        if not query_terms:
            return []
        
        # Get candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        
        # Compute scores for each document
        doc_scores = {}
        for doc_id in candidate_docs:
            score = sum(self._compute_tfidf(term, doc_id) for term in query_terms)
            if score > 0:
                doc_scores[doc_id] = score
        
        # Sort by score descending
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _preprocess_query(self, query_text):
        """
        Preprocess query: tokenize, stem, and remove stopwords
        """
        tokens = query_text.lower()
        pattern = re.compile(r'[a-z]+')
        tokens = pattern.findall(tokens)
        tokens = [self.stemmer.stem(t) for t in tokens if t not in self.stopwords]
        return tokens

if __name__ == "__main__":
    stopwords_path = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_path)
    
    index_path = 'index.txt'
    inverted_index = load_inverted_index(index_path)
    
    # Check if the index.txt file is valid, if not, rebuild the index from the xml file
    meaningful_terms = [term for term in inverted_index.keys() if len(term) >= 3]
    if inverted_index is None or len(meaningful_terms) < 10:
        # Rebuild the index
        text_path = 'collections/trec.5000.xml'
        doc_data = xml_parser(text_path)
        stemmer = PorterStemmer()
        
        for doc in doc_data.keys():
            tokens = preprocess_text(doc_data[doc], stopwords, stemmer)
            doc_data[doc] = tokens
        
        inverted_index = build_inverted_index(doc_data)
        # print the new index to index.txt file
        print_inverted_index(inverted_index)
    
    stemmer = PorterStemmer()
    
    # Boolean retrieval
    boolean_engine = BooleanSearchEngine(inverted_index, stemmer, stopwords)
    boolean_queries = load_queries('queries.boolean.txt')
    
    with open('results.boolean.txt', 'w', encoding='utf-8') as f:
        for i, query in enumerate(boolean_queries, 1):
            results = boolean_engine.process_query(query)
            for doc_id in sorted(results):
                print(f"{i},{doc_id}", file=f)
    
    # Ranked retrieval
    ranked_engine = RankedRetrievalEngine(inverted_index, stemmer, stopwords)
    
    # Load ranked queries
    with open('collections/queries.ranked.txt', 'r', encoding='utf-8') as f:
        ranked_queries = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    ranked_queries.append((int(parts[0]), parts[1]))
    
    # Process ranked queries
    with open('results.ranked.txt', 'w', encoding='utf-8') as f:
        for query_num, query_text in ranked_queries:
            results = ranked_engine.process_query(query_text)
            for doc_id, score in results:
                print(f"{query_num},{doc_id},{score:.4f}", file=f)
