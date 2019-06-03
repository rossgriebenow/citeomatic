from typing import Dict

from torch import tensor, max, clamp, mean
from torch.nn import Module

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.similarity_functions.cosine import CosineSimilarity

#from citeomatic.models.options import ModelOptions
from allennlp.data.vocabulary import Vocabulary

#EmbeddingModel can take either a triplet instance for training the paper embedder or a single query document for inference
#Returns a dict of tensors containing the query embedding and loss (if available)
class EmbeddingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_embedder: TextFieldEmbedder,
                 paper_embedder: Module,
                 pretrained_embeddings=None) -> None:
        
        super().__init__(vocab)
        self.text_embedder = text_embedder
        self.paper_embedder = paper_embedder
        
    def forward(self,
                query_title: Dict[str, tensor],
                query_abstract: Dict[str, tensor],
                query_id: tensor = None,
                candidate_title: Dict[str, tensor] = None,
                candidate_abstract: Dict[str, tensor] = None,
                label: tensor = None) -> Dict[str,tensor]:
        
        query_title_embed = self.text_embedder.forward(query_title["tokens"])
        query_abstract_embed = self.text_embedder.forward(query_abstract["tokens"])
        
        query_embed = self.paper_embedder.forward(query_title_embed, query_abstract_embed)
        
        output = {"query_embed": query_embed}
        
        has_candidate = candidate_title is not None and candidate_abstract is not None
        has_training_examples = has_candidate and label is not None
        
        if has_candidate:
            candidate_title_embed = self.text_embedder.forward(candidate_title["tokens"])
            candidate_abstract_embed = self.text_embedder.forward(candidate_title["tokens"])
            
            candidate_embed = self.paper_embedder.forward(candidate_title_embed, candidate_abstract_embed)
            
            output["cos-sim"] = CosineSimilarity().forward(query_embed,candidate_embed).unsqueeze(-1)
            if has_training_examples:                
                output["loss"] = self._compute_loss(query_embed, candidate_embed, label)
                
        if query_id is not None:
            output["query_id"] = query_id
        
        return output
    
    def _compute_loss(self, query: tensor, candidate: tensor, label: tensor) -> tensor:
        #compute cosine distances between queries and candidates
        #in existing implementation it looks like they're just doing a dot product instead of cosine distance?
        cosine_similarity = CosineSimilarity().forward(query,candidate)
        
        #in existing implementation training examples with even indices are positive/odd indices are negative
        positive = cosine_similarity[::2]
        negative = cosine_similarity[1::2]
        
        #"margin is given by the difference in labels"
        margin = label[::2] - label[1::2]
        delta = clamp(margin + negative - positive,min=0)
        
        return mean(delta)
