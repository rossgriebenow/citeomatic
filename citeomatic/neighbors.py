from typing import Iterable, List

from allennlp.models import Model
from allennlp.data import Instance

from annoy import AnnoyIndex
import tqdm
import logging

class ANN(object):
    def __init__(self, annoy: AnnoyIndex = None, id_dict = None) -> None:
        self.annoy = annoy
        #self.s2_id_to_ann = id_dict

    @staticmethod 
    def build(embedder: Model, instances: Iterable[Instance], vec_size: int, ann_trees: int = 100) -> None:
        logging.info("building annoy index")
        annoy = AnnoyIndex(vec_size)

        batch = list()
        for i, instance in tqdm.tqdm(enumerate(instances)):
            batch.append(instance)
            
            if len(batch) == 64:
                embeddings = embedder.forward_on_instances(batch)
                for embed in embeddings:
                    annoy.add_item(embed["query_id"],embed["query_embed"])
                batch = list()
            
        annoy.build(ann_trees)

        return ANN(annoy)
            
    def save(self, target: str) -> None:
        assert self.annoy is not None
        self.annoy.save('%s.annoy' % target)
        #with open('%s.pickle' % target, 'wb') as handle:
        #    pickle.dump(self.s2_id_to_ann, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def load(source: str, annoy_dims: int) -> object:
        ann = AnnoyIndex(annoy_dims)
        ann.load('%s.annoy' % source)
        #with open('%s.pickle' % source, 'rb') as handle:
        #    id_dict = pickle.load(handle)
        return ANN(ann)
    
    #Annoy returns database indices of nns sorted by distance in increasing order
    def get_nns_by_instance(self, instance: List[Instance],embedder: Model, top_n:int = 50) -> List[int]:
        assert self.annoy is not None
        query_embed = embedder.forward_on_instance(instance)["query_embed"]
        nns = self.annoy.get_nns_by_vector(query_embed,top_n, include_distances=True)
        return nns
    
    def get_nns_by_id(self,idx: int,top_n: int = 50) -> List[int]:
        return self.annoy.get_nns_by_item(idx,top_n)
        
    def get_dist_by_ids(self, i, j) -> float:
        return self.annoy.get_distance(i,j)
