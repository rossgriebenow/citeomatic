from torch import rand, tensor
from torch.nn import Module, Parameter

class Paper_Embedding(Module):
    def __init__(self, pretrained_embeddings=None) -> None:
        super(Paper_Embedding, self).__init__()
        
        self.title_scalar = Parameter(rand(1))
        self.abstract_scalar = Parameter(rand(1))
        
    def forward(self, title: tensor, abstract: tensor) -> tensor:
        title_weighted = title*self.title_scalar
        abstract_weighted = abstract*self.abstract_scalar
        
        return title_weighted+abstract_weighted