import types
from flask import Flask, request
import xml.etree.ElementTree as ET
from transformers import pipeline
import torch
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from transformers import BitsAndBytesConfig

# Example curl call:
#    curl -X POST -d '{"prompt": "How old is the great chinese wall?"}' -H "Content-Type: application/json" http://127.0.0.1:5001

app = Flask(__name__)

def load_config():
    config = types.SimpleNamespace()
    tree = ET.parse("./chat_server_config.xml")
    root_elm = tree.getroot()
    for root_child_elm in root_elm:
        if root_child_elm.tag == 'model_id':
            config.model_id = root_child_elm.text
        elif root_child_elm.tag == 'model_instructions':
            config.model_instructions = root_child_elm.text
        elif root_child_elm.tag == 'port':
            config.port = int(root_child_elm.text)
        elif root_child_elm.tag == 'device':
            config.device = root_child_elm.text
        elif root_child_elm.tag == 'max_new_tokens':
            config.max_new_tokens = int(root_child_elm.text)
        else:
            raise NotImplementedError(f"load_config(): Not implemented config parameter '{root_child_elm.tag}'")
    return config

def messages_to_prompt(messages):  # CF. https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing#scrollTo=lMNaHDzPM68f
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

config = load_config()

# Load the model
"""
pipe = None
if config.model_id == 'HuggingFaceH4/zephyr-7b-alpha':
    pipe = pipeline("text-generation", model=config.model_id, torch_dtype=torch.bfloat16,
                    device_map=config.device)
"""
if config.model_id == 'HuggingFaceH4/zephyr-7b-alpha':
    query_wrapper_prompt = PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {"trust_remote_code": True, 'quantization_config': quantization_config}
    llm = HuggingFaceLLM(
            model_name="HuggingFaceH4/zephyr-7b-alpha",
            tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
            query_wrapper_prompt=query_wrapper_prompt,
            context_window=2048,
            max_new_tokens=config.max_new_tokens,
            model_kwargs=model_kwargs,
            # tokenizer_kwargs={},
            generate_kwargs={"do_sample": False},
            messages_to_prompt=messages_to_prompt,
            device_map="auto"
        )

else:
    raise NotImplementedError(f"chat_server.py: Not implemented model ID '{config.model_id}'")

def Welcome():
    welcome_msg = "Hello! You reached the chat server with a GET call. Send a POST with your prompt to receive a text reply."
    return welcome_msg

@app.route('/', methods=['POST', 'GET'])
def route():
    if request.method == 'GET':
        return Welcome()
    elif request.method == 'POST':
        return reply_to_prompt()

def reply_to_prompt():

    if not 'prompt' in request.json:
        return f"chat_server.reply_to_prompt(): 'prompt' was not found in request.json ({request.json})"
    prompt = request.json['prompt']
    messages = [
        {
            "role": "system",
            "content": config.model_instructions,
        },
        {
            "role": "user",
            "content": prompt},
    ]
    """formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(formatted_prompt, max_new_tokens=config.max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return (outputs[0]["generated_text"])
    """
    response = llm.complete(prompt)
    return str(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.port)