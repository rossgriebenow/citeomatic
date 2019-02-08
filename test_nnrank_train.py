import torch
import numpy  as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer

from typing import Dict

from citeomatic.toydatasetreader import ToyDatasetReader
from citeomatic.testreader import TestReader
from citeomatic.models.text_embedding import Text_Embedding
from citeomatic.models.paper_embedding import Paper_Embedding
from citeomatic.models.embeddingmodel import EmbeddingModel
from citeomatic.models.options import ModelOptions
from citeomatic.models.citationranker import CitationRanker

#just using toydatasetreader to build vocab
reader = ToyDatasetReader()
dataset = reader.read("")
vocab = Vocabulary.from_instances(dataset)

print(vocab.get_vocab_size())

opts = ModelOptions()

reader = TestReader(vocab)
reader.set_compute_nnrank_features(True)
dataset = reader.read("")
text_embedder = Text_Embedding(opts,vocab)

nnrank = CitationRanker(vocab,opts,text_embedder)

iterator = BasicIterator()
iterator.index_with(vocab)

optimizer = torch.optim.SGD(nnrank.parameters(), lr=0.1)

trainer = Trainer(model=nnrank,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=dataset,
                  validation_dataset=dataset,
                  patience=1000,
                  num_epochs=10,
                  summary_interval=2)
                  
trainer.train()