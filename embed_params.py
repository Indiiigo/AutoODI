from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs
from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np

import timeit

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

import ast

# TODO: Clean up and make runnable via parameters

DATAROOT = '../data/'
EMBEDDINGROOT = '../data/embeddings/'


def embed(documents, embedding_type = 'roberta'):
    # define embedding models

       
    if embedding_type == 'roberta':
        model_args = ModelArgs(max_seq_length=156)
        roberta_model = RepresentationModel(
            "roberta",
            "roberta-base",
            args=model_args,
            use_cuda=False
        )

        
        start = timeit.default_timer()
        embeddings = roberta_model.encode_sentences(documents, combine_strategy="mean")
        stop = timeit.default_timer()
    
    elif embedding_type == 'sbert':
        sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        start = timeit.default_timer()
        embeddings = sbert_model.encode(documents)
        stop = timeit.default_timer()
    
    print("time taken for %d: %f" %(len(documents), stop-start))
    return embeddings


def find_similarity(embedded_documents, reference, similarity = 'cosine'):
    sims = []
    for embed in embedded_documents:
        sims.append(1-spatial.distance.cosine(reference, embed))
    return sims

# check an example

"""
# load embeddings
for embedding_type, _ in embedding_types:
	#sustainibility[embedding_type] = list(np.load(EMBEDDINGROOT + 'sustainibility_goals_%s_%s_embeddings.npy' %(text, embedding_type)))
	glassdoor[embedding_type] = list(np.load(EMBEDDINGROOT + 'glassdoor_us_master_%s_%s_embeddings.npy' %(text, embedding_type)))


sents = glassdoor['whole_review_text'].values[0:5]

for sent in sents:
	print()
	print(sent)

print()	

docs_rb = list(embed(sents, roberta_model, 'roberta'))
docs_sb = list(embed(sents, sbert_model, 'sbert'))

print("Similarity with goal 13 \"%s\"" %(sustainibility['Definition'][13]))
print(find_similarity(docs_rb, sustainibility['roberta'][13]))
print(find_similarity(docs_sb, sustainibility['sbert'][13]))

print()
print("Similarity with goal 4 \"%s\"" %(sustainibility['Definition'][4]))
print(find_similarity(docs_rb, sustainibility['roberta'][4]))
print(find_similarity(docs_sb, sustainibility['sbert'][4]))

"""

if __name__ == "__main__":
    # get datasets
    sustainibility = pd.read_csv(DATAROOT + "Sustainability_goals.tsv", sep = "\t")
    glassdoor = pd.read_csv(DATAROOT + "glassdoor_us_master.csv")

    # embed
    # do once and save

    text = 'whole_review_text'
    glassdoor = glassdoor.dropna(subset = [text], axis = 0) # drop nans if that column has it
    
    embedding_types = ['sbert',
				   #'roberta',
				   ]

    for name, data, col in [
						#("sustainibility_goals", sustainibility, 'Definition'), #only need to do once
					    ("glassdoor_us_master", glassdoor, text)]:
        for embedding_type in embedding_types:
            embeddings = list(embed(data[col].values, embedding_type))
            data['embedding_%s' %(embedding_type)] = embeddings
            # save as numpy as pandas seems to corrupt the embeddings
            with open(EMBEDDINGROOT+"/%s_%s_%s_embeddings.npy" %(name, col, embedding_type), 'wb') as f:
                np.save(f, embeddings)
        data.to_csv(DATAROOT + name + "_embedded.csv", sep = '\t', index = False)
