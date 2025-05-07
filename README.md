# Co-Evolving Minds: Self-Play in Multi-Agent LLMs for Advancing Reasoning

Jianing Qi (jq394), Amelia Dai (hd2584), and Tianshu Ni (tn2354)

This repository corresponds to the final project for the Spring 2025 semester of CS 3033 Special Topics: Emerging Topics in Natural Language Processing. Our project introduces a self-play framework for improving mathematical reasoning in large language models by introducing co-evolution between solution generators and solution evaluators.

---

Installation:
Download and install torchtune library. Need to install locally:
```
cd torchtune
pip install -e .
```

Then check the installation:
```
tune ls
```

We should see the recipe of the ```dev/grpo_full_finetune_distributed``` on top of the list. If just install from pip or conda, we will not see the recipe.

Because we have a custom recipe, we need to install it locally not using the offical torchtune.


## SFT
Our SFT data can be found in `sft/data`. The script to prepare the SFT data is in `sft/data/prepare_sft_data.ipynb`.
### Generator SFT
```
tune run lora_finetune_single_device --config ./sft/config/3B_lora_single_device_gen.yaml
```
### Evaluator SFT
```
tune run lora_finetune_single_device --config ./sft/config/3B_lora_single_device_eval.yaml
```
### Mix SFT
```
tune run lora_finetune_single_device --config ./sft/config/3B_lora_single_device_mix.yaml
```

## GRPO Training
This is the official GRPO training on the model
```
tune run --nproc_per_node 4 grpo_distributed --config 3B_grpo_sft.yaml
```

## Selfplay Training
We have two different GRPO based Selfplay training recipes:
1. one to one training
2. one to N training


For the selfplay one to one training:
```
tune run --nproc_per_node 4 dev/selfplay_full_finetune_distributed --config ./3B_full_selfplay_qwen.yaml
```

For the selfplay one to N training:
```
tune run --nproc_per_node 4 dev/evaluator_full_finetune_distributed --config ./3B_full_evaluator_qwen.yaml
```
``` --nproc_per_node``` is the number of GPUs to use.

## Generation
Sepcify the different models in model_name.

### Generator
```
python generate.py \
    --do_generate \
    --model_name selfplay \
    --input_path ./sft/data/gsm8k/gsm8k_test.csv \
    --output_path ./results/final_eval/final_test_selfplay.csv
```

### Evaluator
```
python generate.py \
    --do_evaluate \
    --model_name gen_sft \
    --input_path /scratch/hd2584/llm_marl/results/final_eval/final_test_selfplay.csv \
    --output_path /scratch/hd2584/llm_marl/results/final_eval/final_test_selfplay.csv
```
