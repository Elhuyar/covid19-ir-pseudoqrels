
import pandas as pd
import json
import sys
import os
import random
import argparse

import csv



#metadata="../20200417-trec-rnd1/metadata.csv" #rnd 1     #_covid-19.kwrds.csv.old"
#metadata="../20200502-trec-rnd2/metadata.csv" #rnd 2      #_covid-19.kwrds.csv.old"
#metadata="../20200519-trec-rnd3/metadata.csv" #rnd 3      #_covid-19.kwrds.csv.old"
#metadata="../20200619-trec-rnd4/metadata.csv" #rnd 4      #_covid-19.kwrds.csv.old"
metadata="../20200716-trec-rnd5/metadata.csv" #rnd 3      #_covid-19.kwrds.csv.old"

#qrels="qrels-rnd2.txt"
#qrels="qrels-rnd1.txt"
#qrels="qrels-covid_d3_j2.5-3.txt"
#qrels="qrels-covid_d3_j0.5-3.txt"
#qrels="qrels-covid_d4_j0.5-4.txt"
#qrels="qrels-covid_d5_j0.5-5.txt"
qrels="Silver_title_fullrank_90candidates-rnd5.txt"  #"Silver_title_fullrank_100-1000candidates-rnd5.txt" 

#topic_file="topics-rnd1.tsv"
#topic_file="topics-rnd2.tsv"
#topic_file="topics-rnd3.tsv"
#topic_file="topics-rnd4.tsv"
topic_file="Titles_R5.tsv"

# metadata for documents                                                                                                                                            
metadata_doc=pd.read_csv(metadata,low_memory=False)
sys.stderr.write("metadata shape: {} \n".format(metadata_doc.shape))


topics=pd.read_csv(topic_file,delimiter='\t', header=0)
sys.stderr.write("topics shape: {} \n".format(topics.shape))
#fieldnames=["id", "title"]


rerank_csv="Silver_rnd5_bm25.tsv"
of=open(rerank_csv,"w", encoding='utf-8')
fieldnames=["question", "question_id","answer","answer_id","label"]
wr=csv.DictWriter(of,fieldnames=fieldnames, dialect='excel-tab')


fields_qrels=["id", "q0","doc_id","rank", "score", "run_label"]
bm25ranking=pd.read_csv(qrels,delimiter=' ', header=None, names=fields_qrels)
sys.stderr.write("bm25 ranking shape: {} \n".format(bm25ranking.shape))



def retrieve_example(query,example_row, position,label):
    topic_row=topics[topics["id"]==int(query)]
    if topic_row.empty:
        sys.stderr.write("\r no topic found {} \n ".format(query))
        return {}

    example_row=bm25ranking[bm25ranking['rank'] == position]
    
    if example_row.empty:
        sys.stderr.write("\r no candidate for the given rank found {} \n ".format(position))
        return {}

    candidate = str(example_row.iloc[0]["doc_id"])
    doc_metadata_row = metadata_doc[metadata_doc["cord_uid"]==candidate]

    if doc_metadata_row.empty:
        sys.stderr.write("\r no document metadata for document {} \n ".format(candidate))
        return {}

    #snippet=str(doc_metadata_row.iloc[0]["title"])+" "+str(doc_metadata_row.iloc[0]["abstract"])
    snippet=str(doc_metadata_row.iloc[0]["abstract"])

    #reranking
    q_candidate_id="q-"+str(query)+"-"+str(candidate)
    return {"question":topic_row.iloc[0]["query"],"question_id":query,"answer":snippet,"answer_id":q_candidate_id,"label":label}
     #return {"question":topic_row.iloc[0]["query"]+" "+topic_row.iloc[0]["question"],"question_id":query,"answer":snippet,"answer_id":q_candidate_id,"label":label}
    
    






for i, t in topics.iterrows():
    title_id=t['id']
    sys.stderr.write("current query: \n id= {} \n title: {} ".format(t['id'], t['query']))
    current_query_rank=bm25ranking[bm25ranking["id"] == title_id]

    sys.stderr.write("retrieved ranking for query (shape {}), computing positives...".format(current_query_rank.shape))
    
    #positive example :rank  TOP1
    example=retrieve_example(title_id,current_query_rank, 1,1)
    if bool(example):
        wr.writerow(example)

    sys.stderr.write("retrieved ranking for query, computing negatives...")
    #negative examples: 5 from ranks 20-100
    """
    already={}
    for k in range(0,5):
        count=len(already.keys())
        while count == len(already.keys()):
            rank_pos= random.randint(101,1000)
            if rank_pos not in already.keys():
                already[rank_pos]=1
                example=retrieve_example(title_id,current_query_rank,rank_pos,0)
                if bool(example):
                    wr.writerow(example)
    """
    
of.close()



