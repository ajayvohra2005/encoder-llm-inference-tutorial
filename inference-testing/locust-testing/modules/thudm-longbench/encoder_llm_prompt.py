import datasets
import random

class TextInputGenerator:

    def __init__(self) -> None:
        self.datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        self.dataset = datasets.load_dataset('THUDM/LongBench', random.choice(self.datasets), split='test')
    
    def __call__(self) -> str:
        for example in self.dataset:
            context = example["context"]
            limit = random.randint(16, 150)
            context = " ".join(context.split()[:limit])
            yield context
            
            
    