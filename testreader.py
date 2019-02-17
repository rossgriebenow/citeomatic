from allennlp.data.fields import Field, TextField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary

from torch import tensor

import numpy as np

from typing import Dict, List, Iterator

class ScalarField(Field):
    def __init__(self, scalar: int) -> None:
        super().__init__()
        self.scalar = scalar
        
    def as_tensor(self, padding_lengths: Dict[str, int]) -> tensor:
        return tensor([self.scalar])
        
    def empty_field(self):
        return tensor([])
    
    def get_padding_lengths(self):
        return {}

class TestReader(DatasetReader):
    def __init__(self, vocab: Vocabulary, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=True)
        self.vocab = vocab
        self.vocab_size = self.vocab.get_vocab_size()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.compute_nnrank_features = False
        
    def text_to_instance(self, query_title: List[Token],
                         query_abstract: List[Token],
                         query_id: int = None,
                         candidate_title: List[Token] = None,
                         candidate_abstract: List[Token] = None,
                         label: float = None,
                         candidate_citations: float = None,
                         title_intersection: float = None,
                         abstract_intersection: float = None,
                         similarity: float = None) -> Instance:
        
        query_title_field = TextField(query_title, self.token_indexers)
        query_abstract_field = TextField(query_abstract, self.token_indexers)
        
        fields = {"query_title": query_title_field, "query_abstract": query_abstract_field}
        
        if query_id is not None:
            query_id_field = ScalarField(query_id)
            fields["query_id"] = query_id_field
        
        has_candidate = candidate_title is not None and candidate_abstract is not None
        
        if has_candidate:
            candidate_title_field = TextField(candidate_title, self.token_indexers)
            candidate_abstract_field = TextField(candidate_abstract, self.token_indexers)
            
            fields["candidate_title"] = candidate_title_field
            fields["candidate_abstract"] = candidate_abstract_field
            
        if label is not None:
            fields["label"] = ScalarField(label)
            
        has_nnrank_features = (candidate_citations is not None 
                               and title_intersection is not None 
                               and abstract_intersection is not None 
                               and similarity is not None)
        
        if has_nnrank_features:
            fields["candidate_citations"] = ScalarField(candidate_citations)
            fields["title_intersection"] = ScalarField(title_intersection)
            fields["abstract_intersection"] = ScalarField(abstract_intersection)
            fields["cos_sim"] = ScalarField(similarity)
            
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        #make random data to test models
        for i in range(1000):
            query_title = [Token(self.vocab.get_token_from_index(i)) for i in np.random.randint(2,self.vocab_size,(np.random.randint(5,25)))]
            query_abstract = [Token(self.vocab.get_token_from_index(i)) for i in np.random.randint(2,self.vocab_size,(np.random.randint(50,100)))]
            
            candidate_title = [Token(self.vocab.get_token_from_index(i)) for i in np.random.randint(2,self.vocab_size,(np.random.randint(5,25)))]
            candidate_abstract = [Token(self.vocab.get_token_from_index(i)) for i in np.random.randint(2,self.vocab_size,(np.random.randint(50,100)))]
            
            label = np.random.uniform(0,1)
            
            if self.compute_nnrank_features:
                candidate_citations = np.log(np.random.randint(10,50))
                
                title_intersect = np.random.uniform(0,50)
                abstract_intersect = np.random.uniform(0,50)
                
                cos_sim = np.random.uniform(0,1)
                
                yield self.text_to_instance(query_title,query_abstract, i, candidate_title, candidate_abstract, label, candidate_citations, title_intersect, abstract_intersect, cos_sim)
                
            else:
                yield self.text_to_instance(query_title,query_abstract, i, candidate_title, candidate_abstract, label)
                
    def set_compute_nnrank_features(self, val: bool):
        self.compute_nnrank_features = val