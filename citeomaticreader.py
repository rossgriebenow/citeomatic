from allennlp.data.fields import Field, TextField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary

from torch import tensor

import numpy as np

from typing import Dict, List, Iterator

from citeomatic.neighbors import ANN

import re
from itertools import compress

CLEAN_TEXT_RE = re.compile('[^ a-z]')

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

# filters for authors and docs
MAX_AUTHORS_PER_DOCUMENT = 8
MAX_KEYPHRASES_PER_DOCUMENT = 20

MAX_TRUE_CITATIONS = 100
MIN_TRUE_CITATIONS = 2

# Adjustments to how we boost heavily cited documents.
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02

# Parameters for soft-margin data generation.
TRUE_CITATION_OFFSET = 0.3
HARD_NEGATIVE_OFFSET = 0.2
NN_NEGATIVE_OFFSET = 0.1
EASY_NEGATIVE_OFFSET = 0.0

ANN_JACCARD_PERCENTILE = 0.05

NEG_TO_POS_RATIO = 6

STOPWORDS = {
'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
'the', 'we', 'our', 'which'
}

KEYS = ['hard_negatives', 'nn', 'easy']

margin_multiplier = 1

margins_offset_dict = {
    'true': TRUE_CITATION_OFFSET * margin_multiplier,
    'hard': HARD_NEGATIVE_OFFSET * margin_multiplier,
    'nn': NN_NEGATIVE_OFFSET * margin_multiplier,
    'easy': EASY_NEGATIVE_OFFSET * margin_multiplier
}

column_index = {"index":0,"id":1,"inCitations":2,"outCitations":3,"abstract":4,"title":5}

class CiteomaticReader(DatasetReader):
    def __init__(self, corpus, id_dict, ann, train_frac = 0.95, validation = False, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=True)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.compute_nnrank_features = False
        self.re = re.compile('[^ a-z]')
        self.corpus = corpus
        self.ann = ann
        self.id_dict = id_dict
        self.train_set = int(len(self.corpus)*train_frac)
        self.validation = validation

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
        elif query_id is not None:
            query_id_field = ScalarField(query_id)
            fields["query_id"] = query_id_field

        return Instance(fields)

    def _read(self,str="") -> Iterator[Instance]:
        if self.validation:
            iterator = self.corpus.iloc[self.train_set:].itertuples()
        else:
            iterator = self.corpus.iloc[:self.train_set].itertuples()
            
        for doc in iterator:

            query_title = self._tokenize_text(doc[column_index["title"]])
            query_abstract = self._tokenize_text(doc[column_index["abstract"]])
            query_id = doc[column_index["index"]]

            #get real citations (positive candidate)
            query_citation_ids = [self.id_dict.get(cite) for cite in doc[column_index["outCitations"]]]
            query_citation_ids = list(filter(None,query_citation_ids))
            query_citations = self.corpus.iloc[query_citation_ids]

            #skip documents without enough citations
            if len(query_citation_ids) < MIN_TRUE_CITATIONS:
                continue

            #if document has too many citations take a random subset
            if len(query_citation_ids) > MAX_TRUE_CITATIONS:
                query_citation_ids = np.random.choice(query_citation_ids, MAX_TRUE_CITATIONS, replace=False)

            #get jaccard similarity for all positive examples
            pos_titles = [self._tokenize_text(c) for c in query_citations.title]
            #pos_titles = [self._tokenize_text(db.documents[c].title) for c in query_citation_ids]
            pos_abstracts = [self._tokenize_text(c) for c in query_citations.paperAbstract]
            pos_jaccard_sims = [self._jaccard(query_title,query_abstract, pos_titles[c], pos_abstracts[c]) for c in range(len(query_citation_ids))]

            ann_jaccard_cutoff = np.percentile(pos_jaccard_sims, ANN_JACCARD_PERCENTILE)

            #get negative examples
            N_NEGS_PER_TYPE = int(np.ceil(len(query_citation_ids) * NEG_TO_POS_RATIO / 3.0))

            easy_negs = self._get_easy_negs(query_id, N_NEGS_PER_TYPE, query_citation_ids)
            hard_negs = self._get_hard_negs(query_id, N_NEGS_PER_TYPE, query_citation_ids)
            nn_negs = self._get_nn_negs(query_id, query_title, query_abstract,N_NEGS_PER_TYPE, query_citation_ids, ann_jaccard_cutoff)
            
            n_negs = min([len(easy_negs),len(hard_negs),len(nn_negs)])
            
            shuffle_pos = np.random.randint(0, len(query_citation_ids),size=3*n_negs)
            
            #yield positive and negative examples
            for n, idx in enumerate(shuffle_pos):

                true_citation = query_citation_ids[idx]

                if n % 3 == 0:
                    neg_id = easy_negs.pop()
                    margin = margins_offset_dict["easy"]
                elif n % 3 == 1:
                    neg_id = hard_negs.pop()
                    margin = margins_offset_dict["hard"]
                elif n % 3 == 2:
                    neg_id = nn_negs.pop()
                    margin = margins_offset_dict["nn"]

                if neg_id is not None:
                    try:
                        pos_inst = self._instance_from_ids(query_id,true_citation, margins_offset_dict["true"])
                        neg_inst = self._instance_from_ids(query_id,neg_id, margin)
                        yield pos_inst
                        yield neg_inst
                    except IndexError:
                        continue
                    #yield self._instance_from_ids(query_id,true_citation, margins_offset_dict["true"])
                    #yield self._instance_from_ids(query_id,neg_id, margin)
                else:
                    continue


    def set_compute_nnrank_features(self, val: bool) -> None:
        self.compute_nnrank_features = val

    def _instance_from_ids(self, q: int, c: int, margin: float) -> Instance:
        query = self.corpus.iloc[q]
        candidate = self.corpus.iloc[c]
        query_title = self._tokenize_text(query.title)
        query_abstract = self._tokenize_text(query.paperAbstract)
        candidate_title = self._tokenize_text(candidate.title)
        candidate_abstract = self._tokenize_text(candidate.paperAbstract)
        label = self._cite_boost(len(candidate.inCitations), margin)

        if self.compute_nnrank_features:
            candidate_citations = np.log(len(candidate.inCitations))
            title_intersect = self._single_field_jaccard(query_title, candidate_title)
            abstract_intersect = self._single_field_jaccard(query_abstract, candidate_abstract)
            cos_sim = self.ann.get_dist_by_ids(q, c)
            return self.text_to_instance(query_title, query_abstract, q, candidate_title, candidate_abstract, label, candidate_citations, title_intersect, abstract_intersect, cos_sim)
        else:
            return self.text_to_instance(query_title, query_abstract, q, candidate_title, candidate_abstract, label)


    def _tokenize_text(self, text: str) -> List[Token]:
        tokens = CLEAN_TEXT_RE.sub(' ', text.lower()).split()
        mask = [word not in STOPWORDS for word in tokens]
        return [Token(word) for word in compress(tokens,mask)]

    def _jaccard(self, x_title, x_abstract, y_title, y_abstract) -> float:
        x_title = [str(t) for t in x_title]
        x_abstract = [str(t) for t in x_abstract]
        y_title = [str(t) for t in y_title]
        y_abstract = [str(t) for t in y_abstract]

        a = set(x_title + x_abstract)
        b = set(y_title + y_abstract)
        c = a.intersection(b)
        if len(a)+len(b) == len(c):
            return 0
        else:
            return float(len(c)) / (len(a) + len(b) - len(c))

    def _single_field_jaccard(self, x, y) -> float:
        a = set([str(t) for t in x])
        b = set([str(t) for t in y])
        c = a.intersection(b)
        if len(a)+len(b)==len(c):
            return 0
        else:
            return float(len(c)) / (len(a) + len(b) - len(c))


    def _cite_boost(self, citations, offset):
        sigmoid = 1 / (1 + np.exp(-citations * CITATION_SLOPE))
        return offset + (sigmoid * MAX_CITATION_BOOST)

    def _get_easy_negs(self, query_id: int, n: int, pos: List[int]) -> List[int]:
        ids = list()
        while len(ids) < n:
            idx = np.random.randint(0,self.train_set)
            if idx not in pos and idx not in ids:
                ids.append(idx)
        return ids

    def _get_hard_negs(self, query_id: int, n: int, pos: List[int]) -> List[int]:
        ids = list()
        my_pos = pos.copy()
        while len(ids) < n and len(my_pos) > 0:
            rand_pos = np.random.choice(my_pos)
            citations = [self.id_dict.get(cite) for cite in self.corpus.iloc[rand_pos].outCitations]
            citations = list(filter(None, citations))
            if len(citations) > 0:
                need_citation = True
                while need_citation:
                    idx = np.random.choice(citations)
                    if idx not in pos and idx not in ids and idx != query_id:
                        ids.append(idx)
                        need_citation = False
                    else:
                        citations.remove(idx)
                        
                    if len(citations) == 0:
                        my_pos.remove(rand_pos)
                        need_citation = False
            else:
                my_pos.remove(rand_pos)
                
        return ids


    def _get_nn_negs(self, query_id: int, query_title: List[Token], query_abstract: List[Token], n: int, pos: List[int], cutoff: float) -> List[int]:
        ids = list()
        try:
            nns = self.ann.get_nns_by_id(query_id,10*n)
        except IndexError:
            return []
        i = 0
        while len(ids) < n and i < 10*n:
            candidate = self.corpus.iloc[nns[i]]
            candidate_title = self._tokenize_text(candidate.title)
            candidate_abstract = self._tokenize_text(candidate.paperAbstract)
            jaccard = self._jaccard(query_title, query_abstract, candidate_title, candidate_abstract)
            if jaccard < cutoff and nns[i] not in pos and nns[i] != query_id:
                ids.append(nns[i])
            i += 1
        return ids


class SimpleReader(DatasetReader):    
    def __init__(self, corpus, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=True)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.re = re.compile('[^ a-z]')
        self.corpus = corpus
        
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
        query_id_field = ScalarField(query_id)
        
        fields = {"query_title": query_title_field, "query_abstract": query_abstract_field, "query_id": query_id_field}            
        return Instance(fields)
    
    def _read(self,str="") -> Iterator[Instance]:
        for row in self.corpus.itertuples():
            query_title = self._tokenize_text(row[column_index["title"]])
            query_abstract = self._tokenize_text(row[column_index["abstract"]])
            query_id = row[column_index["index"]]
            
            yield self.text_to_instance(query_title, query_abstract, query_id)
            
    def _tokenize_text(self, text: str) -> List[Token]:
        tokens = CLEAN_TEXT_RE.sub(' ', text.lower()).split()
        mask = [word not in STOPWORDS for word in tokens]
        return [Token(word) for word in compress(tokens,mask)]
