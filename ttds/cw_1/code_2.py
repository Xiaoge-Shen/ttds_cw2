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

if __name__ == "__main__":
    stopwords_path = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_path)
    text_path = 'collections/trec.5000.xml'
    doc_data = xml_parser(text_path)
    # print(doc_data)
    stemmer = PorterStemmer()
    # for doc in doc_data.keys():
    #     tokens = preprocess_text(doc_data[doc], stopwords, stemmer)
    #     doc_data[doc] = tokens
    # print(doc_data)
    inverted_index = build_inverted_index(doc_data)
    print("done")
    print_inverted_index(inverted_index)
    # print(inverted_index)