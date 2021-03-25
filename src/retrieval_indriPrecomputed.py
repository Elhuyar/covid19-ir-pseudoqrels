#import pyndri
from nltk.tokenize import RegexpTokenizer

import pandas as pd
import json
import sys
import os
import random
import argparse

import csv


from math import exp

def process_results(indri_results,metadata_df, metadata_pas_df, reranking_scores_df, query_id, coord_type, rerank_weight, rerank_cutoff, max_rank_docs, passages=False):
    output=[]
    count=0

    #score normalization
    min=1
    max=0
    for index, p in indri_results.iterrows():
        score=p['score']
        if exp(score) < min:
            min=exp(score)
        if exp(score) > max:
            max=exp(score)

   
    #coordinates normalization
    """
    x_df = metadata_df[coord_type+"_coord_x"]
    if passages :
        x_df = metadata_pas_df[coord_type+"_coord_x"] 

    min_coord_x=1
    max_coord_x=0
    for i,x in x_df.iteritems():
        if x <  min_coord_x:
            min_coord_x=x
        if x > max_coord_x:
            max_coord_x=x
            
    y_df = metadata_df[coord_type+"_coord_y"]
    if passages :
        y_df = metadata_pas_df[coord_type+"_coord_y"] 

    min_coord_y=1
    max_coord_y=0
    for i,y in y_df.iteritems():
        if y <  min_coord_y:
            min_coord_y=y
        if y > max_coord_y:
            max_coord_y=y
    """
            
    #loop throgout result and prepare output
    for index, p in indri_results.iterrows():
        count+=1
        if count > max_rank_docs:
            sys.stderr.write("maximum number of documents reached the rest of the ranking will be discarded (count =  {}).\n".format(count))
            break
            
        score=p['score']
        ext_document_id = p['doc']
        int_document_id=p['rank']
        doc_id = ext_document_id
        sys.stderr.write("\r processed {} documents/passages {} ".format(count, ext_document_id))
        snippet=""
        coords = {"coord_x":random.uniform(0, 1),"coord_y":random.uniform(0, 1)}
        

        
        if passages == True:
            passage_metadata_row = metadata_pas_df[metadata_pas_df["paragraph_id"]==int(ext_document_id)]
            if passage_metadata_row.empty:
                sys.stderr.write("\r no passage metadata for document {} \n ".format(ext_document_id))
                continue

            doc_id=passage_metadata_row.iloc[0]["cord_uid"]
            snippet=passage_metadata_row.iloc[0]["text"]


        # common fields for documents and passages
        doc_metadata_row = metadata_df[metadata_df["cord_uid"]==doc_id]
        if doc_metadata_row.empty:
            sys.stderr.write("\r no document metadata for document {} \n ".format(ext_document_id))
            continue
        url=doc_metadata_row.iloc[0]["url"]
        title=doc_metadata_row.iloc[0]["title"]
        author=doc_metadata_row.iloc[0]["authors"]
        journal=doc_metadata_row.iloc[0]["journal"]
        publish_date=doc_metadata_row.iloc[0]["publish_time"]

        ##normalized coordinates. If there is no coordinates information coordinates are left random
        
        if coord_type+"_coord_x" in doc_metadata_row and coord_type+"_coord_x" in passage_metadata_row:
            coord_x = (doc_metadata_row.iloc[0][coord_type+"_coord_x"]-min_coord_x)/(max_coord_x-min_coord_x)
            coord_y = (doc_metadata_row.iloc[0][coord_type+"_coord_y"]-min_coord_y)/(max_coord_y-min_coord_y) 
            if passages == True:
                coord_x = (passage_metadata_row.iloc[0][coord_type+"_coord_x"]-min_coord_x)/(max_coord_x-min_coord_x)
                coord_y = (passage_metadata_row.iloc[0][coord_type+"_coord_y"]-min_coord_y)/(max_coord_y-min_coord_y) 
            coords = {"coord_x":coord_x,"coord_y":coord_y}
        
        
        #reranking
        q_candidate_id="q-"+str(query_id)+"-"+str(doc_id)
        if passages == True:
            q_candidate_id = q_candidate_id+"_"+ext_document_id
            
        bert_score=None #bert score, if not found do not take it into account
        reranking_score_row=reranking_scores_df[reranking_scores_df["query_candidate_id"]==q_candidate_id]
        if not reranking_score_row.empty:
            bert_score=reranking_score_row.iloc[0]["pos_score"]
            #sys.stderr.write("bert score for candidate {}: {} \n".format(q_candidate_id,bert_score))
            
        indri_score=(exp(score)-min)/(max-min)  # normalized indri score

        """
        if bert_score != None:
            ranking_score=(1-rerank_weight)*indri_score+(rerank_weight*bert_score)
        else:
            ranking_score=indri_score
        """

            
        ranking_score=indri_score
        if bert_score != None:
            if rerank_cutoff <= 0:
                ranking_score=(1-rerank_weight)*indri_score+(rerank_weight*bert_score)
            else:
                sys.stderr.write("cutoff =  {}; count =  {}".format(rerank_cutoff, count))
                if count <= rerank_cutoff :        #if passages == True and bert_score != None:
                    ranking_score=(1-rerank_weight)*indri_score+(rerank_weight*bert_score)
                else:
                    sys.stderr.write("cutoff surpassed (count =  {}) from here on indri scores are left as they are.\n".format(count))
        
        
        if passages == False:
            snippet=str(doc_metadata_row.iloc[0]["title"])+" "+str(doc_metadata_row.iloc[0]["abstract"])
            #snippet=doc_metadata_row.iloc[0]["abstract"]
    

        #generate uniq doc_ids for both pas and docs
        if passages == True:
            doc_id= doc_id+"_"+ext_document_id
        
        doc ={"doc_id":doc_id, "title":title, "journal":journal,"author":author,"publish_date":publish_date, "url":url,"text":snippet,"ranking_score":ranking_score, "indri_score":indri_score, "rank":int_document_id, "coordinates": coords}
        output.append(doc)
        #print(ext_document_id, score)

    return output
    
##################################################
##                  Main function            #####
##                                           #####
##################################################       
def main(args):

    ## command line arguments
    queries=args.queries
    maxdocs=args.maxdocs
    metadata_path=args.metadata_path
    index_root=args.index_path
    reranking_scores=args.reranking_scores
    coord_type=args.coordinates_algorithm
    krovetz_stem=args.krovetz_stem
    stopword_file=args.stopwords
    no_rerank = args.no_rerank
    rerank_weight=args.rerank_weight
    rerank_cutoff=args.rerank_cutoff
    
    #metadata="metadata.csv_covid-19.kwrds.csv.all-coords.csv"
    #passage_metadata="metadata.csv_covid-19.kwrds.paragraphs.csv.all-coords.csv"
    metadata="metadata.csv" #_covid-19.kwrds.csv.old"
    passage_metadata="metadata.csv_covid-19-empty.kwrds.paragraphs.csv"    #"metadata.csv_covid-19.kwrds.paragraphs.csv"
    
    # metadata for documents
    metadata_doc=pd.read_csv(os.path.join(metadata_path,metadata),dtype={"mag_id":str,"who_covidence_id":str,"arxiv_id":str})
    sys.stderr.write("metadata shape: {} \n".format(metadata_doc.shape))

    # if passages are to be retrieved instead of full documents open also metadata for passages.
    metadata_pas=pd.read_csv(os.path.join(metadata_path,passage_metadata))
    sys.stderr.write("metadata shape: {} \n".format(metadata_pas.shape))


    reranking_scores_df=pd.DataFrame(columns=["query_candidate_id","label","neg_score","pos_score"])
    # if exists, reranking-scores file
    if os.path.isfile(reranking_scores):
        reranking_scores_df=pd.read_csv(reranking_scores,dialect='excel-tab')


    rerank_csv="rerank-queries_nofilter-precomp.tsv"
    of=open(rerank_csv,"w", encoding='utf-8')
    fieldnames=["question", "question_id","answer","answer_id","label"]
    wr=csv.DictWriter(of,fieldnames=fieldnames, dialect='excel-tab')
    #wr.writeheader()


    # output format for bokeh
    output=[]
    documents=[]
    passages=[]
    #fieldnames=["doc_id","source","author", "url","title",]

    # indri
    #index_doc_path=os.path.join(index_root,'BildumaTRECAbsBodyIndex_ezmarra')#_ round 1')
    #index_doc_path=os.path.join(index_root,'BildumaTRECAbsBodyIndex_2ndround')
    #index_doc_path=os.path.join(index_root,'BildumaTRECAbsIndex_round3_all') #BildumaTRECAbsIndex_round3')
    #index_doc_path=os.path.join(index_root,'BildumaTRECAbsIndex_round3all_exp') #BildumaTRECAbsIndex_round3all_exp')
    #index_doc_path=os.path.join(index_root,'BildumaTRECAbsIndex_round4_Nofiltered')

    indri_df=pd.read_csv(index_root, sep=' ', header=None, names=["topic","Q0","doc","rank","score","run_id"])
    
    queries_df = pd.read_csv(queries,dialect='excel-tab')
    for index, row in queries_df.iterrows(): 
        #querylc = row['query'].lower()
        querylc = row['question'].lower()+" "+row['query'].lower() #+" "+row['narrative'].lower()
        querylc2 = row['narrative'].lower()
        
        sys.stderr.write("current query: {} -- {}\n.".format(querylc,querylc2))

        # document level results from file
        indri_query=indri_df[indri_df["topic"]==row['id']]
        
        docs = process_results(indri_query,metadata_doc, metadata_pas, reranking_scores_df, row["id"], coord_type,rerank_weight, rerank_cutoff, maxdocs)

        #sys.stderr.write("docs retrieved, {} \n".format(len(docs)))

        for d in docs:
            #wr.writerow({"question":row["query"],"question_id":row["id"],"answer":d["text"],"answer_id":d["doc_id"],"label":0})
            #wr.writerow({"question":row["question"],"question_id":row["id"],"answer":d["text"],"answer_id":d["doc_id"],"label":0})
            wr.writerow({"question":row["query"]+" "+row["question"],"question_id":row["id"],"answer":d["text"],"answer_id":d["doc_id"],"label":0})
            
        
        # passage level results
        #results = prf_query_env.query(tokenized_query, results_requested=maxdocs)
        #pas = process_results(results,index_pas,metadata_doc, metadata_pas, reranking_scores_df, row["id"], coord_type, rerank_weight, passages=True)

        pas_df = pd.DataFrame(docs)

        pas_sorted = pas_df.sort_values("ranking_score",ascending=False)
        if no_rerank :
            pas_sorted = pas_df.sort_values("indri_score",ascending=False)
        
        #sys.stderr.write("passages retrieved, {} \n".format(len(pas)))

        doc_dict={}
        rank=1
        for index, p in pas_sorted.iterrows():
            #wr.writerow({"question":row["query"]+" "+row["narrative"],"question_id":row["id"],"answer":p["text"],"answer_id":p["doc_id"],"label":0})
            
            #sys.stderr.write(" {} - {} {} \n".format(p["doc_id"]))
            
            #ranking has already 1000 documents
            if rank > maxdocs:
                break
            
            question_id=row["id"]
            #doc_pas_id = str(p["doc_id"])
            doc_id = str(p["doc_id"])
            #doc_id = doc_pas_id.split("_")[0]

            # already found a more relevant passage of the same document
            if doc_id in doc_dict:
                continue
            
            doc_dict[doc_id]=1
            if no_rerank:
                print("{} Q0 {} {} {} {}".format(row['id'],doc_id, rank, p["indri_score"],"elhuyar_indri"))
            else:
                print("{} Q0 {} {} {} {}".format(row['id'],doc_id, rank, p["ranking_score"],"elhuyar_rRnk"))
            #sys.stderr.write("{} Q0 {} {} {} {}\n".format(row['id'],doc_id, rank, p["ranking_score"],run_rerank))
            rank+=1

        #query_json={"query_id":row['id'], "task": row['task'], "query":row['query'], "docs":docs,"pas":pas}

        #query_json={"query_id":row['id'], "task": "trec-round-1", "query":row['query'], "pas":pas}
        #output.append(query_json)



    of.close()    
    #print(json.dumps(output, indent=4, sort_keys=True))



##################################################
##              parameter parsing            #####
##################################################        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='script returns document and passage level result for a given indri collection (two indexes, docs and passages), inplemented for kaggle-covid-19 challenge.',
        epilog="type python3 -u retrieval.py -h for help",
        prog='retrieval.py' )

    parser.add_argument("queries", type=argparse.FileType('r'), help="File containing queries for document or or passage retrieval. tsv format, including one column called 'query'.")  
    parser.add_argument("-i", "--index-path", type=str, default='/media/nfs/multilingual/kaggle-covid19/xabi_scripts', help="path to the file containing indri rankings")
    parser.add_argument("-m", "--metadata-path", type=str, default='/media/nfs/multilingual/kaggle-covid19', help="path to the folder containing metadata files")
    parser.add_argument("-r", "--reranking-scores", type=str, default='/media/nfs/multilingual/kaggle-covid19/reranking_scores.tsv', help="file containing scores from the finetuned BERT for reranking)")
    parser.add_argument("-rw", "--rerank_weight", type=float, default=0.2, help="Weight of the reranking scores)")
    parser.add_argument("-rc", "--rerank_cutoff", type=int, default=0, help="rerank only the first n documents of the ranking. Default is 0, meaning the whole ranking will be reranked.")
    parser.add_argument("-c", "--coordinates-algorithm", type=str, choices=['fasttext', 'tfidf'], default='fasttext', help="Algorithm used for computing document and passage coordinates, defaults to fasttext)")
    parser.add_argument("-d", "--maxdocs", type=int, default=50, help="max number of results to return (default is 50)")
    parser.add_argument("-k", "--krovetz_stem", action='store_true', help="Apply Krovetz stemmer to queries.")
    parser.add_argument("-n", "--no-rerank", action='store_true', help="Whether indri-based results should be returned instead of reranked results. For testing purposes.")
    parser.add_argument("-s", "--stopwords", type=str, default='/media/nfs/multilingual/kaggle-covid19/covid-19-IR/resources/stopwords-en.txt', help="file containing stopwords for the tokenizer and indri querying)")


    args=parser.parse_args()

    #check if test_file was provided
    if args.queries is None:
        sys.stdout.write("no queries supplied ")
        exit
        
    #if args.embeddings is None:
    #    args.embedding_update=False;
            
    sys.stderr.write(str(args).replace(', ','\n\t')+"\n")
    sys.stderr.flush()
    main(args)

