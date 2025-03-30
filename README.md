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


## Fine tuning
```
tune run --nproc_per_node 4 full_finetune_distributed --config 3B_grpo_sft.yaml
```

## GRPO Training
```
tune run --nproc_per_node 4 grpo_distributed --config 3B_grpo_sft.yaml
```

## generation
```
tune run generate --config ./custom_generation_config.yaml
```
