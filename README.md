

# Learning Source Code Vulnerability Localization and Causation with Explainability


The widespread adoption of smart connected IoT devices
underscores the need to address code vulnerabilities to protect secu-
rity and privacy. These code vulnerabilities can cause unauthorized ac-
cess, data breaches, network disruptions, and privacy violations. In this
paper, we address the challenge of identifying source code vulnerabil-
ities by offering clear code vulnerability indicators and causation. In
our system, we create the code property graph (CPG) of the entire
source repository. We perform a thorough analysis by extracting func-
tions from the source code and classifying vulnerable functions. Subse-
quently, we pass the vulnerable functions through an ensemble trans-
former and graph model to pinpoint the vulnerability’s location and
provide the root cause of the existence of the vulnerability through ex-
plainability. We demonstrate the effectiveness of our proposed system
by detecting 24 N-day and 3 zero-day vulnerabilities by analyzing six
IoT repositories, including TinyOS, Contiki, Zephyr, FreeRTOS, RIOT-
OS, and Raspberry Pi OS from GitHub




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




### Datasets
- Please download our [Dataset](https://drive.google.com/drive/folders/1zmTpSvyyC9usKiMu-kuEZHbF0_UqDeKw?usp=sharing) .




## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
