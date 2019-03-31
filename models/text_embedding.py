from typing import Dict

from torch import tensor, sum
from torch.nn import Dropout
from torch.nn.functional import normalize

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

class Text_Embedding(TextFieldEmbedder):
    def __init__(self, vocab: Vocabulary, dense_dim=75, l2=1e-5, l1=1e-7, drop=0.1)-> None:
        super(Text_Embedding, self).__init__()
        
        self.dense_dim = dense_dim
        self.dropout_p = drop
        self.l1_lambda = l1
        self.l2_lambda = l2
        self.final_l2_norm = True
        
        self.embed_direction = Embedding(num_embeddings = vocab.get_vocab_size('tokens'), 
                                         embedding_dim = self.dense_dim, norm_type = 2,
                                         max_norm = self.l2_lambda)
        self.embed_magnitude = Embedding(num_embeddings = vocab.get_vocab_size('tokens'),
                                         embedding_dim = 1,
                                         norm_type = 1,
                                         max_norm = self.l1_lambda)
        
        #pytorch hasn't implemented spatial dropout for 1d
        self.dropout = Dropout(p = self.dropout_p)
        
    def forward(self, text_field_input: Dict[str, tensor], num_wrapping_dims: int = 0)-> tensor:
        direction = self.embed_direction.forward(text_field_input)
        direction_normalized = normalize(direction,p=2,dim=-1)
        
        magnitude = self.embed_magnitude.forward(text_field_input)
        embedding = direction_normalized*magnitude

        if self.final_l2_norm:
            summed = sum(embedding,dim=-2)
            normalized_sum = normalize(summed,p=2,dim=-1)
            return self.dropout.forward(normalized_sum)
        else:
            summed = sum(embedding,dim=-2)
            return self.dropout.forward(summed)        
        
    def get_output_dim(self) -> int:
        return self.dense_dim