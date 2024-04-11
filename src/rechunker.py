import glob
from llmsherpa.readers import LayoutPDFReader
import pandas as pd

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

class Rechunker:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def read_pdf(self, pdf_url):
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(pdf_url)
        return doc
    
    def clean_doc(self, doc):
        doc = [x.encode('ascii', 'ignore').decode('ascii') for x in doc if len(x)>100]
        return doc
    
    def check_criteria(self, lst):
        if lst[-1] == ":" or "below" in "\n".join(lst):
            return True
        else:
            return False
    
    def process_doc(self, doc):
        prev_tag = ""
        doc_sentences = list(doc["sentences"])
        doc_tags = list(doc["tag"])
        merge_list, j = [], -1
        
        for i in range(len(doc)):
            # print (i, prev_tag, doc_tags[i])
            if doc_tags[i] == "para" and prev_tag == "para" and self.check_criteria(merge_list[j]):
                merge_list[j].extend(doc_sentences[i])
                prev_tag = "para"
                
            elif doc_tags[i] == "list_item" and prev_tag == "para":
                merge_list[j].extend(doc_sentences[i])
                prev_tag = "list_item"
                
            elif doc_tags[i] == "list_item" and prev_tag == "list_item":
                merge_list[j].extend(doc_sentences[i])
                prev_tag = "list_item"
                
            elif doc_tags[i]=="para":
                merge_list.append(doc_sentences[i])
                prev_tag = "para"
                j = j + 1
            else:
                prev_tag = doc_tags[i]
        return merge_list
        
    def get_paras(self):
        directory_path = self.pdf_path
        all_data = []
        filenames = []
        search_pattern = f"{directory_path}/*.pdf"
        for i in glob.glob(search_pattern):
            print (i)
            doc = self.read_pdf(i)
            doc = pd.DataFrame(doc.json)
            doc = self.process_doc(doc)
            doc = ["\n".join(x) for x in doc]
            doc = self.clean_doc(doc)
            filenames.extend([i]*len(doc))
            all_data.append(doc)
            print (len(all_data), len(filenames))
        return all_data, filenames