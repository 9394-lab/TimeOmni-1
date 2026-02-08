import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Download TimeOmni-1 model")
    parser.add_argument("--model", type=str, default="anton-hugging/TimeOmni-1-7B")
    args = parser.parse_args()

    AutoTokenizer.from_pretrained(args.model)
    AutoModelForCausalLM.from_pretrained(args.model)
    print("done")


if __name__ == "__main__":
    main()