import chainlit as cl
import logging
import subprocess
import requests
import xml.etree.ElementTree as ET
import types

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

config = None

@cl.on_chat_start
async def on_chat_start():
    logging.info("on_chat_start()")

    global config

    # Load the configuration file
    config = load_configuration_file("./app_config.xml")
    logging.info(f"config = \n{config}")

    # Check if the service is up
    url_responds = False
    try:
        command = f"curl {config.llm_service_url}"
        result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, check=True)
        url_responds = True
    except subprocess.CalledProcessError:
        url_responds = False
    if not url_responds:
        logging.error(f"The llm service at address {config.llm_service_url} doesn't respond")


@cl.on_message
async def main(message: cl.Message):
    logging.info(f"main({message}); message.content = {message.content}")

    global config
    
    prompt = f"{config.prompt_seed}\n{message.content}"
    data_dict = {"prompt": prompt}
    session = requests.Session()
    result = session.post(
        config.llm_service_url,
        json=data_dict
    )
    if result.status_code != 200:
        logging.error(f"chat_with_service.main(): The result status code ({result.status_code}) is not 200")

    # Find the beginning of the useful message
    response = trim_response(result.text, config.start_of_response_marker)

    # Send a response to the user
    await cl.Message(
        content=response,
    ).send()

def trim_response(response, start_of_response_marker):
    start_of_marker = response.find(start_of_response_marker)
    if start_of_marker == -1:  # The marker was not found
        return response
    start_of_message = start_of_marker + len(start_of_response_marker)
    return response[start_of_message:]

def load_configuration_file(configuration_filepath):
    logging.info("load_configuration_file()")
    config = types.SimpleNamespace()
    config_doc = ET.parse(configuration_filepath)
    root_elm = config_doc.getroot()
    for root_child_elm in root_elm:
        if root_child_elm.tag == 'LLMServiceUrl':
            config.llm_service_url = root_child_elm.text
        elif root_child_elm.tag == 'StartOfResponseMarker':
            config.start_of_response_marker = root_child_elm.text
        elif root_child_elm.tag == 'PromptSeed':
            config.prompt_seed = root_child_elm.text
        else:
            raise NotImplementedError(f"load_configuration_file(): Not implemented configuration element <{root_child_elm.tag}>")
    return config