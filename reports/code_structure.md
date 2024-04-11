## Folder structure

--pdfs: raw pdfs

--raw_text: text extracted by adobe, utilized for table extraction

-- utils.utils: utilities like flatten_list used across modules

-- logs: correct and in-correct predictions

-- test.xlsx: sample pdf extracted by llm_sherpa

-- main: main function for sentence transformer, calls several other modules created in src

-- data: datafiles with extracted paras and tables from llm_sherpa

-- src.post_processing: removing retrieval if below a hard threshold, caters to no answer

-- src.eval: MRR and Recall@k

-- src.encoder: contains encoding functions for tf-idf, sentence transformer and dragon

-- src.faiss: contains flatindex 

-- output: output of level 1 and level 2, note this done not None the results 

-- scripts: scripts for dragon, splade, tf-idf that import classes * functions from src, utils etc 

-- document_questions: annotated with ground truth and tags
