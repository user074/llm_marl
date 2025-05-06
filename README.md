# llm_marl

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


## Fine tuning
```
tune run --nproc_per_node 4 full_finetune_distributed --config 3B_grpo_sft.yaml
```

## GRPO Training
This is the official GRPO trainign on the model
```
tune run --nproc_per_node 4 grpo_distributed --config 3B_grpo_sft.yaml
```

## generation
```
tune run generate --config ./custom_generation_config.yaml
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


