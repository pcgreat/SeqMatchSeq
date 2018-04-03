# SeqMatchSeq (PyTorch)
Pytorch Implementations of the model described in the papers related to sequence matching:

- [A Compare-Aggregate Model for Matching Text Sequences](https://arxiv.org/abs/1611.01747) by Shuohang Wang, Jing Jiang

Note:
* Originally forked from [here](https://github.com/shuohangwang/SeqMatchSeq), which is the author's code implementation in **Torch**
* Cleaned up a little bit, and reimplemented in **PyTorch**
* Only reimplemented WikiQA tasks, other tasks are not supported yet.
* Author's original repo reaches **0.734 (MAP)** in Wikiqa Dev, and this code reaches **0.727 (MAP)**
* Due to differences in API between torch and pytorch, you might notice some differences:
    * The orig repo calculates the gradient explicitly, while this one uses **[optim](http://pytorch.org/docs/master/optim.html)** for auto grad.
    * The orig repo uses TemporalConvolution, and this one uses nn.Conv1d in pytorch
    * I am not sure why orig repo uses Parallel, so I just removed it
* TODO:
    * The save/load module function needs to be fixed
    * Best_params needs to be kept after each evaluation
    * Integrate into ParlAI, and test on InsuranceQA as well
    


### Requirements
- [Pytorch v0.1.12](http://pytorch.org/)
- tqdm
- Python 3

### Datasets
- [InsuranceQA Corpus V1: Answer Selection Task](https://github.com/shuzi/insuranceQA)
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

For now, this code only support SNLI and WikiQA data sets.

### Usage

WikiQA task:
```
sh preprocess.sh wikiqa (Please first dowload the file "WikiQACorpus.zip" to the path SeqMatchSeq/data/wikiqa/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419)

PYTHONPATH=. python3 main/main.py --task wikiqa --model compAggWikiqa --comp_type mul --learning_rate 0.004 --dropoutP 0.04 --batch_size 10 --mem_dim 150

- `model` (model name) : compAggWikiqa 
- `comp_type` (8 different types of word comparison): only mul works for now, but you can easily add others and send me pull request
```
