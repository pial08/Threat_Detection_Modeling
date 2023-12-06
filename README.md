

# Causative Insights into Open Source Software Security using \\ Large Language Code Embeddings and Semantic Vulnerability Graph

Open Source Software (OSS) security and resilience are worldwide phenomena hampering economic and technological innovation. OSS vulnerabilities can cause unauthorized access, data breaches, network disruptions, and privacy violations, rendering any benefits worthless. While recent deep-learning techniques have shown great promise in identifying and localizing vulnerabilities in source code, it is unclear how effective these research techniques are from a usability perspective due to a lack of proper methodological analysis of how beneficial these are to the end users. Usually, these methods offload a developer's task of classifying and localizing vulnerable code; still, a reasonable study to measure the actual effectiveness of these systems has yet to be conducted. To address the challenge of proper developer training from the prior methods, we propose a system to link vulnerabilities to their root cause, thereby intuitively educating the developers to code more securely. Furthermore, we provide a comprehensive usability study to test the effectiveness of our system in fixing vulnerabilities and its capability to assist developers in writing more secure code. We demonstrate the effectiveness of our system by showing its efficacy in helping developers fix source code with vulnerabilities. Our study shows a 24\% improvement in code repair capabilities compared to previous methods. We also show that, when trained by our system, on average, approximately 9\% of the developers naturally tend to write more secure code with fewer vulnerabilities. 

## System Architecture

[](https://github.com/pial08/Threat_Detection_Modeling/blob/main/figures/t5-gcn.png)
![alt text](https://github.com/pial08/Threat_Detection_Modeling/blob/main/figures/t5-gcn.png?raw=true)


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
