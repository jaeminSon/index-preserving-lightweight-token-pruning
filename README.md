# Simple Yet Efficient Token Pruning for Document Understanding in Vision-Language Models
[arXiv (PDF)](https://arxiv.org/abs/YYMM.NNNNN)


## Generate a Qwen2_5 Model with the Pruning capability

```
1. Download Qwen2.5 huggingface model
2. Copy the downloaded Qwen2.5 directory (say this copied directory 'P')
3. Modify 'architectures' and 'auto_map' in config.json in the directory 'P' like qwen2_5_7b/config.json
4. Copy qwen2_5_7b/modeling_qwen2_5_vl.py to the directory 'P'
5. Run merge_classifier_weights.py
```


## Run merge_classifier_weights.py

```
$ python3 merge_classifier_weights.py --source <path/to/the/downloaded/directory> --path_model <path/to/directory/'P'/modeling_qwen2_5_vl.py>  --classifier_weights <path/to/classifier_weights>
```
