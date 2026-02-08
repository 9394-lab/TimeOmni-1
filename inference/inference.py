import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "Output Format:\n"
    "<think>Your step-by-step reasoning process that justifies your answer</think>\n"
    "<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>"
)


def build_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Single-prompt inference")
    parser.add_argument("--model_dir", type=str, default="anton-hugging/TimeOmni-1-7B")
    parser.add_argument("--question", type=str, default="You are given two time series related to river discharge measurements, expressed in m^3/s. Through causal discovery methods, we aim to identify potential causal relationships between different measuring stations from time-series data alone. The time series of J96A is: [3.35, 2.92, 2.61, 2.92, 4.48, 7, 15.71, 10.65, 7.16, 5.79, 5.42, 5.31, 5, 4.38, 3.87, 3.52, 3.21, 2.92, 2.51, 2.39, 2.21, 2.08, 1.9, 1.75, 1.62, 1.56, 1.43, 1.4, 1.31, 1.24, 1.24, 1.25, 0.96, 2.75, 2.54, 2.03, 2.27, 2.36, 2.24, 2, 2.36, 2.16, 2.4, 2.11, 2.04, 1.96, 2.35, 2.26, 2.45, 2.19, 2.15, 1.91, 1.8, 1.64, 1.53, 1.44], The time series of UC1U is [5.19, 4.52, 4.2, 4.45, 6.29, 8, 22, 12.66, 8.48, 7.51, 7.15, 7.24, 7.42, 6.85, 6.24, 5.75, 5.37, 4.84, 4.45, 4.24, 3.94, 3.72, 3.57, 3.29, 3.12, 3, 2.85, 2.74, 2.68, 2.59, 2.56, 2.49, 2.55, 4.37, 4.71, 3.58, 3.84, 3.88, 3.8, 3.44, 3.64, 3.46, 3.57, 3.34, 3.19, 3.03, 3.39, 3.24, 3.45, 3.19, 3.14, 3, 2.86, 2.66, 2.56, 2.47]. Please identify the causal relationships between the two measurement stations? The data is collected every 12 hours from 2021-02-01 to 2021-02-28 totally 56 points each series.\n\n\nOptions:\nA. UC1U is the cause and J96A is the effect.\nB. J96A is the cause and UC1U is the effect.\nC. J96A and UC1U are not causal.")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    inputs = tokenizer(build_prompt(args.question, args.system_prompt), return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
