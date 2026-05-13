import re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer

def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f}
        return stopwords
    except FileNotFoundError:
        return set()  

def preprocess_text(text, stopwords, stemmer):
    tokens = text.lower()
    pattern = re.compile(r'[a-z]+')
    tokens = pattern.findall(tokens)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

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

def build_inverted_index(doc_data):
    inverted_index = {}
    for i, tokens in doc_data.items():
        for position, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if i not in inverted_index[token]:
                inverted_index[token][i] = []
            inverted_index[token][i].append((position))
    return inverted_index

def print_inverted_index(inverted_index):
    with open('index.txt', 'w', encoding='utf-8') as f:
        for token, positions in inverted_index.items():
            df = len(positions)
            print(f"{token}:{df}", file=f)

            for i, position in positions.items():
                positions = ','.join([str(pos) for pos in position])
                print(f"\t{i}: {positions}", file=f)

def load_inverted_index(filepath):
    index = {}
    current_term = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('\t'):
                    term, df = line.strip().split(':')
                    current_term = term
                    index[current_term] = {}
                else:
                    doc_id, positions = line.strip().split(':')
                    doc_id = int(doc_id)
                    positions = [int(pos) for pos in positions.strip().split(',')]
                    index[current_term][doc_id] = positions
    except Exception as e:
        return None
    return index

def load_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f]
    return queries

def preprocess_query(query, stemmer):
    query = query.split(':')[-1].strip()
    query = query.lower()
    query = stemmer.stem(query)
    return query

def preprocess_single_term(term, stopwords, stemmer):
    # term = term.lower().strip()
    term = stemmer.stem(term)
    return term
def process_query(query, inverted_index):
    single_term_pattern = re.compile(r'^[a-z]+$')
    single_term_match = single_term_pattern.match(query)

    phrase_pattern = re.compile(r'^"(.*)"$')
    phrase_match = phrase_pattern.match(query)

    if single_term_match:
        return inverted_index[query]
    elif phrase_match:
        print(query)
        # term1, term2 = [term.strip('"') for term in query]
        term1, term2 = query.strip('"').split()
        term1 = stemmer.stem(term1)
        term2 = stemmer.stem(term2)
        if term1 in inverted_index and term2 in inverted_index:
            return process_phrase(term1, term2, inverted_index)
    elif query == 'AND':
        return {}
    elif query == 'OR':
        return {}
    elif query == 'NOT':
        return {}
    else:
        return {}
    
def process_phrase(term1, term2, inverted_index):
    result = {}
    index1 = inverted_index[term1]
    index2 = inverted_index[term2]
    for doc_id in index1:
        for position in index1[doc_id]:
            if doc_id not in index2:
                continue
            if position + 1 in index2[doc_id]:
                if doc_id not in result:
                    result[doc_id] = []
                result[doc_id].append(position)
    return result

class BooleanSearchEngine:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index
    
    def process_query(self, query):
        # 1. 解析查询类型
        query_type = self._identify_query_type(query)
        
        # 2. 根据类型处理
        if query_type == "single_term":
            return self._handle_single_term(query)
        elif query_type == "or_query":
            return self._handle_or_query(query)
        elif query_type == "and_not_query":
            return self._handle_and_not_query(query)
        elif query_type == "phrase_query":
            return self._handle_phrase_query(query)
        elif query_type == "proximity_query":
            return self._handle_proximity_query(query)
        elif query_type == "complex_query":
            # return self._handle_complex_query(query)
            pass
    
    def _handle_single_term(self, query):
        return self.inverted_index[query]
    
    def _handle_or_query(self, query):
        return self.inverted_index[query]
    
    def _handle_and_not_query(self, query):
    def _identify_query_type(self, query):
        # 识别查询类型
        if re.match(r'^[a-zA-Z]+$', query.strip()):
            return "single_term"
        elif ' OR ' in query:
            return "or_query"
        elif ' AND NOT ' in query:
            return "and_not_query"
        elif query.startswith('"') and query.endswith('"'):
            return "phrase_query"
        elif re.match(r'#\d+\(', query):
            return "proximity_query"
        else:
            return "complex_query"

if __name__ == "__main__":
    stopwords_path = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_path)
    text_path = 'collections/trec.5000.xml'
    doc_data = xml_parser(text_path)
    stemmer = PorterStemmer()
    for doc in doc_data.keys():
        tokens = preprocess_text(doc_data[doc], stopwords, stemmer)
        doc_data[doc] = tokens

    # inverted_index = build_inverted_index(doc_data)

    index_path = 'index.txt'
    inverted_index = load_inverted_index(index_path)

    queries_path = 'queries.boolean.txt'
    queries = load_queries(queries_path)
    engine = BooleanSearchEngine(inverted_index)
    for query in queries:
        processed_query = engine.process_query(query)
        print(processed_query)
    # for i in range(len(queries)):
    #     query = queries[i]
    #     query = preprocess_query(query, stemmer)
    #     processed_query = process_query(query, inverted_index)
    #     print(processed_query)
    #     if processed_query == None:
    #         continue
    #     for j in range(len(processed_query)):
    #         doc_id = list(processed_query.keys())[j]
    #         positions = processed_query[doc_id]
    #         for position in positions:
    #             print(f"{doc_id},{position}")
        


