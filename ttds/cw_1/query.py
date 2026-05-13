import re

def load_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f]
    return queries

def preprocess_query(query):
    query = query.split(':')[1].strip()
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    return query
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
if __name__ == "__main__":
    query_path = 'queries.boolean.txt'
    queries = load_queries(query_path)
    inverted_index_path = 'index.txt'
    inverted_index = load_inverted_index(inverted_index_path)
    processed_queries = []
    for query in queries:
        query = query.split(':')[1].strip()
        processed_queries.append(query)
        # query = preprocess_query(query)
        # processed_queries.append(query)
    print(processed_queries)

        # print(inverted_index[query])
    # print(inverted_index)
    # print(inverted_index['ft'].keys())
