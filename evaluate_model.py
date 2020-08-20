import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import regex as re
import argparse

def load_model_and_tokenizer(model_path: str, device: str):
  tokenizer = YTEncoder.from_pretrained(model_path)

  model = GPT2LMHeadModel.from_pretrained(model_path)
  model.to(device)
  model.eval()
  return (model, tokenizer)

def get_sample(model, tokenizer, device, prompt: str, length:int, num_samples:int, temperature: int = 1.0, top_k: int = 0, top_p = 0.9, allow_linebreak:bool = True):
    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [1] + tokenizer.encode('[')[-1:] + tokenizer.encode('(')[-1:]
    filter_single += [] if allow_linebreak else filter_n

    context_tokens = tokenizer.encode(prompt)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        filter_single=filter_single,
        filter_double=filter_n,
        num_samples=num_samples,
    ).to('cpu')

    prompt = tokenizer.decode(context_tokens)
    len_prompt = len(prompt)
   
    replies = [out[item, :].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item)[len_prompt:] for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in text]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, text)]
    return result

def print_sample(samples):
  for index, sample in enumerate(samples):
      print(sample)
      print(f"-------SAMPLE {index} END-------")

def continuous_run(model, tokenizer, device, args):
  while True:
    prompt = input("Prompt: ")
    results = get_sample(model, tokenizer, device, prompt, args.length, args.num_samples, args.temperature, args.top_k, args.top_p, True)
    print_sample(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Model path")
    parser.add_argument("--continuous_run", action="store_true",
                        help="Prompt for input continuously.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    if args.continuous_run:
      continuous_run(model, tokenizer, device, args)
    else:
      results = get_sample(model, tokenizer, device, args.prompt, args.length, args.num_samples, args.temperature, args.top_k, args.top_p, True)
      print_sample(results)

if __name__ == "__main__":
  main()