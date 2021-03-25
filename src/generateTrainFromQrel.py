
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
qrels="qrels-covid_d5_j0.5-5.txt"

#topic_file="topics-rnd1.tsv"
#topic_file="topics-rnd2.tsv"
#topic_file="topics-rnd3.tsv"
#topic_file="topics-rnd4.tsv"
topic_file="topics-rnd5.tsv"

# metadata for documents                                                                                                                                            
metadata_doc=pd.read_csv(metadata,low_memory=False)
sys.stderr.write("metadata shape: {} \n".format(metadata_doc.shape))


topics=pd.read_csv(topic_file,delimiter='\t')
sys.stderr.write("topics shape: {} \n".format(topics.shape))
fieldnames=["id", "query","question", "narrative"]


rerank_csv="rerank-queries-from-qrels.tsv"
of=open(rerank_csv,"w", encoding='utf-8')
fieldnames=["question", "question_id","answer","answer_id","label"]
wr=csv.DictWriter(of,fieldnames=fieldnames, dialect='excel-tab')




with open(qrels) as f:
    for line in f:
        fields=line.split()
        rnd=fields[1]
        candidate=fields[2]
        query=fields[0]
        label=int(fields[3].strip())

        topic_row=topics[topics["id"]==int(query)]
        if topic_row.empty:
            #sys.stderr.write("\r no topic found {} \n ".format(query))
            continue

        doc_metadata_row = metadata_doc[metadata_doc["cord_uid"]==candidate]
        if doc_metadata_row.empty:
            sys.stderr.write("\r no document metadata for document {} \n ".format(candidate))
            continue

        snippet=str(doc_metadata_row.iloc[0]["title"])+" "+str(doc_metadata_row.iloc[0]["abstract"])


        #reranking
        q_candidate_id="q-"+str(query)+"-"+str(candidate)

        #if label < 2:
        wr.writerow({"question":topic_row.iloc[0]["query"]+" "+topic_row.iloc[0]["question"],"question_id":query,"answer":snippet,"answer_id":q_candidate_id,"label":label})


of.close()
