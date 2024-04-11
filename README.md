## Information Retrieval - Approach

- What do you think about your solution and how do you think it can be scaled up?
Have a document classifier depending on what information we have available 
Better chunking strategies 
: Have a segmentation model? or something else
Metrics for multi-document retrieval 
Better labelled data to figure out those metrics 
Different kinds of indexes e.g. HNSW, other ANN based techniques to reduce retrieval latency (just comapred L2 and cosine distance)


- What have you already done?
Problem Exploration:
First I looked at the data to see what kind of questions were being asked 
Discovered that some questions had no answers e.g. What are the standard rules of a golf game?
and majority of the questions could be answered using a single paragraph
Also noted that one of the PDFs had no raw text. Moreover the PDFs were text based and not image based. 
Then I looked at the raw text and decided to setup a basic pipeline as my baseline

Based on this I decided the basic problems I would be targeting to solve:
0) What metrics to use
1) Which db/index to use 
2) Which encoder to use
3) Re-chunking: Some problems in raw text which required context across two chunks and the second chunk would not be retrieved without the first context
4) Tabular data: Some questions were based on tables and the tabular data was extracted pretty well from the documents 
5) How to handle no answer? ~ would the LLM handle this? I think it woud

Solutions: Based just on para based data

0a) Metrics: Recall@4 and looked at the results as well as the paper "Lost in the middle" to see how much context can an LLM handle without missing information (although this was true for documents it was a good proxy) - Recall@10
0b) Metrics: Earlier contexts are better and the paper argues that also in the end so I also decided to have a metric which captures the rank  (MRR); giving higher preference to Recall and then to MRR; If I had more question which require multi-document retrieval, I would have also considered NDCG (but it essentially a scaled version of MRR if there is only 1 item to be retrieved)  

1) Index: Milvus support is excellent for Linux but not good for Windows so I decided to use FAISS. Start with a flatIndex, using L2 distance of cosine distance. Exretemely fast so I decided not to move on to indexes like HNSW or other Approximate nearest neighbors based indexes 

2a) Sentence Transformers: Worked on a previous problem where they worked well so I used that as a starting point. Trained using Siamese tower based approach so I was aware that they're better for sentence level representations rather than paragraph level representations 

~ 29%


Error Analysis:
50% errors in para based retrieval were due to chunking and 50% due to fetching the wrong contexts


Solution1: Solve chunking
LayoutPDFReader  

Solution2: Use other encoders like TF-IDF and dragon 
Reason why TF-IDF worked and dragon did not: Exact phrase matching is poor in dense retreivers
