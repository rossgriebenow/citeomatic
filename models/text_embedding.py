from typing import Dict

from torch import tensor, sum
from torch.nn import Dropout
from torch.nn.functional import normalize

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from citeomatic.models.options import ModelOptions

class Text_Embedding(TextFieldEmbedder):
    def __init__(self, options: ModelOptions, vocab: Vocabulary, pretrained_embeddings=None, final_l2_norm=True)-> None:
        super(Text_Embedding, self).__init__()
        
        self.dense_dim = options.dense_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.enable_fine_tune = options.enable_fine_tune
        self.dropout_p = options.dropout_p
        self.l1_lambda = options.l1_lambda
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.final_l2_norm = True
        
        #postpone implementing pretrained embeddings, might be unneccessary if allennlp can handle everything
        trainable=self.pretrained_embeddings is None or self.enable_fine_tune
        
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