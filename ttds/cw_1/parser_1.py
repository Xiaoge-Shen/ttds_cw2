import xml.etree.ElementTree as ET

# def parse_xml(filepath):
#     tree = ET.parse(filepath)
#     root = tree.getroot()
#     return root

# if __name__ == "__main__":
#     filepath = 'collections/sample.xml'
#     root = parse_xml(filepath)
#     print(root)

def parser(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    for doc in root:
        doc_data = {}
        docno = doc[0].text
        headline = doc.findtext('HEADLINE', '')
        text = doc.findtext('TEXT', '')
        doc_data['docno'] = docno
        doc_data['content'] = (headline + ' ' + text).strip()
        print(doc_data)
if __name__ == "__main__":
    filepath = 'collections/trec.sample.xml'
    # filepath = 'collections/sample.xml'
    root = parser(filepath)