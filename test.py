print("importing...")

from typing import Dict

from citeomatic.toydatasetreader import ToyDatasetReader

from citeomatic.models.text_embedding import Text_Embedding
from citeomatic.models.paper_embedding import Paper_Embedding
from citeomatic.models.embeddingmodel import EmbeddingModel
from citeomatic.models.options import ModelOptions

print("imported citeomatic modules...")
#and for this demo we need:
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer

#build dataset, vocabulary, reader
reader = ToyDatasetReader()
dataset = reader.read("")
vocab = Vocabulary.from_instances(dataset)

#build text_embedder
opts = ModelOptions()
text_embedder = Text_Embedding(opts,vocab)

#test text_embedder
x = reader.read("")[0]["query_title"]
x1 = reader.read("")[0]["query_abstract"]
print(x.index(vocab))
x1.index(vocab)
p_lengths = x.get_padding_lengths()
p1_lengths = x1.get_padding_lengths()
print(p_lengths)
print(p_lengths.values())
x_t = x.as_tensor(dict.fromkeys(p_lengths,max(p_lengths.values())))
x1_t = x1.as_tensor(dict.fromkeys(p1_lengths,max(p_lengths.values())))
print(x_t)

title_embed = text_embedder.forward(x_t["tokens"])
abs_embed = text_embedder.forward(x1_t["tokens"])
print(title_embed)



#build paper_embedder
paper_embedder = Paper_Embedding()

#build iterator
iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)


#build embedding model and "train"
embedder = EmbeddingModel(vocab, text_embedder, paper_embedder)

optimizer = torch.optim.SGD(embedder.parameters(), lr=0.1)

trainer = Trainer(model=embedder,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=dataset,
                  validation_dataset=dataset,
                  patience=1000,
                  num_epochs=10,
                  summary_interval=2)

try:
	trainer.train()
except RuntimeError:
	print("model needs to be passed candidate and label to train, but feedforward works!")

print(list(embedder.parameters()))