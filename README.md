# CoF-CoT
This repository provides evaluation datasets, implementation and sample demos for the paper [*CoF-CoT: Enhancing Large Language Models with Coarse-to-Fine Chain-of-Thought Prompting for Multi-domain NLU Tasks*](https://arxiv.org/abs/2310.14623) **(EMNLP'2023 Main Conference)**

## Support
We provide support for API Calls from 2 Large Language Models (LLMs), including [PaLM](https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html) and [GPT3.5-turbo](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates).
Please make sure you have your own valid API keys and update them in ```key.txt``` and ```google_key.txt``` files correspondingly before running the experiments.

## Prerequisites
Refer to documentation of PaLM](https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html) and [GPT3.5-turbo](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates) regarding pre-requisites.

## Dataset
We conduct evaluations on subsets of [MTOP dataset](https://aclanthology.org/2021.eacl-main.257/) and [MASSIVE dataset](https://aclanthology.org/2023.acl-long.235/).
Both evaluation datasets are few-shot in-context learning samples are provided. Please refer to our paper for further details.


## Running Experiments/ Demonstrations
Please refer to the manuscript regarding the detailed rationale of the experiment design.
Individual prompts contain minor updates to account for the generated outputs.
```
bash run_query.sh ${dataset} ${model_type}
```

where passing arguments ${.} are defined as follows: 
* ```dataset```: Evaluation Dataset (i.e. MASSIVE or MTOP)
* ```model_type```: Backbone LLMs (i.e. palm or gpt)


## Citation
If you find our ideas, code or dataset helpful, please consider citing our work as follows:
<pre>
@article{nguyen2023cof,
  title={CoF-CoT: Enhancing Large Language Models with Coarse-to-Fine Chain-of-Thought Prompting for Multi-domain NLU Tasks},
  author={Nguyen, Hoang H and Liu, Ye and Zhang, Chenwei and Zhang, Tao and Yu, Philip S},
  journal={arXiv preprint arXiv:2310.14623},
  year={2023}
}
</pre>

