from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import pickle
import argparse

# temp_path = '/scratch/hd2584/llm_marl/sft/data/gpt-4o/gsm8k_train_eval.pkl'

BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)

EVALUATION_PROMPT = (
    "%s. "
    "The Assistant evaluates the problem and the solution. The assistant first evaluates and finds out where the solution is wrong. "
    "Then the assistant provides the verification. "
    "The evaluation process and verification are enclosed within <evaluate></evaluate> and <verify></verify> tags, respectively, "
    "i.e., <evaluate>evaluation process here</evaluate> <verify>verification here</verify>. Verification is limited to 'correct' or 'wrong'. Assistant:<evaluate>"
)

def load_model(model_name):
    if 'gpt' in model_name:
        return OpenAI()
    else:
        if model_name == 'base':
            llm = LLM(
                    model="Qwen/Qwen2.5-3B",
                    download_dir="/scratch/hd2584/llm_marl/model/Qwen/Qwen2.5-3B",
                    dtype="float16")
        elif model_name == 'mix_sft':
            llm = LLM(
                    model = 'ameliadai/qwen_3b_mix_sft',
                    download_dir = '/scratch/hd2584/llm_marl/model/ameliadai/qwen_3b_mix_sft',
                    dtype="float16")
        elif model_name == 'eval_sft':
            llm = LLM(
                    model = '/scratch/hd2584/llm_marl/model/qwen_3b_eval_sft/epoch_9',
                    dtype="float16")
        elif model_name == 'grpo':
            llm = LLM(
                    model = 'user074/grpo_qwen3b',
                    download_dir='/scratch/hd2584/llm_marl/model/user074/grpo_qwen3b',
                    dtype="float16")
        elif model_name == 'selfplay':
            llm = LLM(
                    model = 'user074/selfplay_qwen3b_mix_SFT',
                    download_dir='/scratch/hd2584/llm_marl/model/user074/selfplay_qwen3b_mix_SFT',
                    dtype="float16")
        elif model_name == 'selfplay_nosft_one_to_one':
            llm = LLM(
                    model = 'user074/selfplay_qwen3b',
                    download_dir='/scratch/hd2584/llm_marl/model/user074/selfplay_nosft_one_to_one',
                    dtype="float16")
        elif model_name == 'selfplay_nosft':
            llm = LLM(
                    model = 'user074/selfplay_qwen3b_evaluator',
                    download_dir='/scratch/hd2584/llm_marl/model/user074/selfplay_qwen3b_evaluator',
                    dtype="float16")
        elif model_name == 'gen_sft':
            llm = LLM(
                    model='/scratch/hd2584/llm_marl/model/qwen_3b_gen_sft/epoch_9',
                    dtype="float16")

        return llm

def model_generate(llm, model_name, prompts):
    res_ls = []
    if 'gpt' in model_name:
        idx = 0
        for prompt in tqdm(prompts):
            idx += 1
            prompt = [{"role": "system", "content": ""},
                    {"role": "user", "content": prompt}]
            response = llm.chat.completions.create(
                            model=model_name,
                            messages=prompt,
                            max_tokens=1024
                        )
            generated_text = response.choices[0].message.content
            print(generated_text)
            res_ls.append(generated_text)

            if (idx+1) % 50 == 0:
                with open(temp_path, "wb") as f:
                    pickle.dump(res_ls, f)
    else:
        sampling_params = SamplingParams(
            max_tokens=1024,
            # temperature=1, # for regeneration in multi-round setting
            temperature=0,
            top_p=0.9,
        )
        for output in llm.generate(prompts, sampling_params=sampling_params):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_ls.append(generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return res_ls

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="run evaluation.")
    parser.add_argument('--do_generate', action='store_true', help='Run generation')
    parser.add_argument('--do_evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-3B', help='Name of the model to use')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--temp_path', type=str, default=None, help='Optional path to save temporary pickle file')

    args = parser.parse_args()

    llm = load_model(args.model_name)

    if args.do_generate:
        print('--- Do generate...')
        data = pd.read_csv(args.input_path)
        prompts = [BASE_PROMPT % question for question in data['question']]
    elif args.do_evaluate:
        print('--- Do evaluate...')
        data = pd.read_csv(args.input_path)
        data['gen_prompt'] = [BASE_PROMPT % question for question in data['question']]
        for idx, row in data.iterrows():
            res = row['gen_prompt'] + row['response']
            data.loc[idx, 'eval_prompt'] = EVALUATION_PROMPT % res
        prompts = data['eval_prompt']

    res_ls = model_generate(llm, args.model_name, prompts)

    if args.do_generate:
        data['response'] = res_ls
        data.to_csv(args.output_path, index=False)
    elif args.do_evaluate:
        data['evaluate'] = res_ls
        data.to_csv(args.output_path, index=False)