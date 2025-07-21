

# Supporting Entry-Level Developers in Secure Coding: Explainable Root-Cause Analysis for Vulnerability Remediation

Software vulnerability detection tools are commonly utilized in software development, but their effectiveness for entry-level programmers, particularly in vulnerability remediation, remains understudied. Novice programmers, typically recent computer science graduates, often lack sufficient experience in identifying and mitigating code vulnerabilities. Therefore, understanding their specific challenges is critical to enhancing the security of software developed by entry-level professionals. To explore this issue, we conducted an empirical study with a sample size of 56 participants, divided into multiple groups, drawn from a population of approximately 6,000 Computer Science students at a tier-one university. Participants implemented simple programs assessed for functional correctness and security vulnerabilities, resulting in a vulnerability incidence rate of 72.1%. When using state-of-the-art vulnerability detection tools, the incidence dropped modestly to 56.2%. Subsequent surveys revealed that inadequate root-cause explanations significantly impeded remediation effectiveness. Addressing these shortcomings, we introduce the Vulnerability Root-Cause Analyzer (VulRCA), integrating DeepLiftSHAP-based root-cause analysis into a classification framework leveraging T5 and Graph Convolutional Networks (GCN). VulRCA explicitly pinpoints vulnerable code segments to aid novice developers in effective remediation. Our evaluation demonstrates an  18.2% improvement in vulnerability remediation using VulRCA, highlighting its potential to substantially improve entry-level developers' secure coding practices.

## System Architecture



#### Requirements
- Python 	3.7
- Pytorch 	1.9 
- Transformer 	4.4
- torchmetrics 0.11.4
- tree-sitter 0.20.1
- sctokenizer 0.0.8

Moreover the above libraries can be installed by the commands from *requirements.txt* file. It is assumed that the installation will be done in a Linux system with a GPU. If GPU does not exist please remove the first command from the *requirements.txt*  file and replace it with 

`conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch` for OSX

or 


`conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch` for Linux and Windows with no GPU.

Instructions to install libraries using *requirements.txt* file.

```shell
cd code 
pip install -r requirements.txt
```


### Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).





Please run the following commands:

```shell
cd lineloc

./run.sh

or,

python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --do_sorting_by_line_scores \
  --learning_rate=6e-6 \
  --weight_decay=0.9 \
  --epochs=25 \
  --effort_at_top_k=0.9 \
  --top_k_recall_by_lines=0.01 \
  --top_k_recall_by_pred_prob=0.2 \
  --reasoning_method=all \
  --train_data_file=../data/D2A_Dataset/train.csv \
  --eval_data_file=../data/D2A_Dataset/val.csv \
  --test_data_file=../data/D2A_Dataset/test.csv \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 

```





## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
