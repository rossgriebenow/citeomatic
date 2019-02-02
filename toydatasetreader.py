from typing import Dict, List, Iterator

from allennlp.data import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

#For training we'll need a DatasetReader producing instances of the form {"query_title":,"query_abstract":, "candidate_title":, "candidate_abstract":, "label:"}
#For inference we'll take instances of the form {"query_title":,"query_abstract"}

class ToyDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
        self.data=[["Acute spastic entropion.","Experiments were performed to determine whether growth hormone-releasing hormone (GHRH) mRNA displays diurnal variations in the hypothalamus and cortex of the rat. Levels of GHRH and beta-actin mRNA were measured from hypothalamic and cortical extracts using the reverse transcriptase polymerase chain reaction method in rats sacrificed at 4 h intervals across a 12:12 h light:dark cycle. Hypothalamic GHRH mRNA peaked around the light onset, declined during the light period, and stayed low in the dark. Variations in hypothalamic beta-actin and cortical GHRH mRNA levels were not observed. beta-Actin mRNA expression in the cortex was higher in the dark than in the light period. The results demonstrate that hypothalamic GHRH mRNA displays diurnal variations."],["Primary Debulking Surgery Versus Neoadjuvant Chemotherapy in Stage IV Ovarian Cancer","Primary debulking surgery (PDS) has historically been the standard treatment for advanced ovarian cancer. Recent data appear to support a paradigm shift toward neoadjuvant chemotherapy with interval debulking surgery (NACT-IDS). We hypothesized that stage IV ovarian cancer patients would likely benefit from NACT-IDS by achieving similar outcomes with less morbidity. Patients with stage IV epithelial ovarian cancer who underwent primary treatment between January 1, 1995 and December 31, 2007, were identified. Data were retrospectively extracted. Each patient record was evaluated to subclassify stage IV disease according to the sites of tumor dissemination at the time of diagnosis. The Kaplan–Meier method was used to compare overall survival (OS) data. A total of 242 newly diagnosed stage IV epithelial ovarian cancer patients were included in the final analysis; 176 women (73%) underwent PDS, 45 (18%) NACT-IDS, and 21 (9%) chemotherapy only. The frequency of achieving complete resection to no residual disease was significantly higher in patients with NACT-IDS versus PDS (27% vs. 7.5%; P < 0.001). When compared to women treated with NACT-IDS, women with PDS had longer admissions (12 vs. 8 days; P = 0.01), more frequent intensive care unit admissions (12% vs. 0%; P = 0.01), and a trend toward a higher rate of postoperative complications (27% vs. 15%; P = 0.08). The patients who received only chemotherapy had a median OS of 23 months, compared to 33 months in the NACT-IDS group and 29 months in the PDS group (P = 0.1). NACT-IDS for stage IV ovarian cancer resulted in higher rates of complete resection to no residual disease, less morbidity, and equivalent OS compared to PDS."]]
        
    def text_to_instance(self, title: List[Token], abstract: List[Token],) -> Instance:
        title_field = TextField(title, self.token_indexers)
        abstract_field = TextField(abstract, self.token_indexers)
        fields = {"query_title": title_field, "query_abstract": abstract_field}

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        for paper in self.data:
            title_list = paper[0].strip().split()
            abs_list = paper[1].strip().split()
            print(self.text_to_instance([Token(word) for word in title_list],[Token(word) for word in abs_list]))
            yield self.text_to_instance([Token(word) for word in title_list],[Token(word) for word in abs_list])