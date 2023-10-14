# Cf. https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
import logging
import torch
from transformers import pipeline
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    prompt,
    instructions
):
    logging.info(f"zephyr.main()\n\tprompt: '{prompt}'\n\tinstructions: {instructions}")

    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', help="The prompt")
    parser.add_argument('--instructions', help="The system instructions. Default: 'You are a polite, tongue-in-cheek funny friend'",
                        default='You are a polite, tongue-in-cheek funny friend')
    args = parser.parse_args()
    main(
        args.prompt,
        args.instructions
    )