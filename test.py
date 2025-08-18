from nnsight import LanguageModel
from datasets import load_dataset

model = LanguageModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
dataset = load_dataset("walledai/AdvBench", split="train")
chats = [
    {"role": "user", "content": prompt}
    for prompt in dataset["prompt"]
]

templated_prompts = model.tokenizer.apply_chat_template(chats, tokenize=False)
print(templated_prompts)



