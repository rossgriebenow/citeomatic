import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import tqdm

from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
import torch
import torch.optim as optim

from citeomatic.models.text_embedding import Text_Embedding
from citeomatic.models.paper_embedding import Paper_Embedding
from citeomatic.models.embeddingmodel import EmbeddingModel
from citeomatic.models.citationranker import CitationRanker
from citeomatic.citeomaticreader import CiteomaticReader, SimpleReader
from citeomatic.neighbors import ANN
from citeomatic.eval_metrics import eval_text_model

def from_pkl(n_files):
    df_list = list()
    for i in tqdm.tqdm(range(n_files)):
        #print(i)
        df_list.append(pd.read_pickle("../clean_pickle/df0"+str(i)+".pkl"))

    return pd.concat(df_list)
    
#load data
print("loading data into memory")
df = from_pkl(1)

print("building index dictionary")
idx_to_id_dict = {}
for row in tqdm.tqdm(df.itertuples()):
    idx_to_id_dict[row[1]] = row[0]

print("initializing model")

vocab = Vocabulary.from_files("../vocabulary")

text_embedder = Text_Embedding(vocab)
paper_embedder = Paper_Embedding()
embedder = EmbeddingModel(vocab, text_embedder, paper_embedder)

rank_abs_embed = Text_Embedding(vocab)
rank_title_embed = Text_Embedding(vocab)
ranker = CitationRanker(vocab,rank_title_embed,rank_abs_embed)

ann = ANN.load("../bigger_ann",75)

train_frac = 0.995
train_set = int(len(df)*train_frac)

ann_reader = SimpleReader(df)
ann_data = ann_reader.read("")

#val_reader = CiteomaticReader(df,idx_to_id_dict,ann,train_frac=train_frac, validation = True)
#val_reader.set_compute_nnrank_features(True)

simple = SimpleReader(df.iloc[train_set:])
val_data = list(simple.read(""))

iterator = BasicIterator(batch_size=16)
iterator.index_with(vocab)

n_epochs=40

optimizer = optim.Adam(embedder.parameters(), lr=0.001)

if torch.cuda.is_available():
    cuda_device = 0
    embedder = embedder.cuda(cuda_device)
    ranker = ranker.cuda(cuda_device)
else:
    cuda_device = -1

#allennlp doesn't have callbacks so we call the trainer one epoch at a time
print("beginning training")
for e_i in range(n_epochs):
    #(re)build ann
    #print("bulding annoy trees")
    #ann = ANN.build(embedder, ann_data, vec_size=text_embedder.get_output_dim(), ann_trees=10)

    #check validation metrics
    print("Evaluating model performance...")
    val_reader = CiteomaticReader(df,idx_to_id_dict,ann,train_frac=train_frac, validation = True)
    val_reader.set_compute_nnrank_features(True)
    valid = eval_text_model(val_data,val_reader,embedder,ann,ranker,df, idx_to_id_dict)
    print(valid)
    
    #make new reader with the new ann
    embed_training_reader = CiteomaticReader(df,idx_to_id_dict,ann,train_frac=train_frac, validation = False)
    embed_training_reader.set_compute_nnrank_features(False)
    embed_training_data = embed_training_reader.read("")

    rank_training_reader = CiteomaticReader(df,idx_to_id_dict,ann,train_frac=train_frac, validation = False)
    rank_training_reader.set_compute_nnrank_features(True)
    rank_training_data = embed_training_reader.read("")
    
    embed_trainer = Trainer(model=embedder,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=embed_training_data,
                      #validation_dataset=val_data,
                      patience=10,
                      num_epochs=1,
                      shuffle=False,
                      cuda_device=cuda_device)

    rank_trainer = Trainer(model=ranker,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=rank_training_data,
                      #validation_dataset=val_data,
                      patience=10,
                      num_epochs=1,
                      shuffle=False,
                      cuda_device=cuda_device)
    
    #run for an epoch
    print("Beginning Embedder Epoch")
    embed_trainer.train()
    
    print("Beginning NNRank Epoch")
    rank_trainer.train()
    
    #make checkpoint
    print("Making model checkpoint")
    with open("/chkpt/embedder_e"+str(e_i)+".th", 'wb') as f:
        torch.save(embedder.state_dict(), f)
        
    with open("/chkpt/ranker_e"+str(e_i)+".th", 'wb') as f:
        torch.save(ranker.state_dict(), f)
