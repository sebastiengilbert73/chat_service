# chat_service
A service to chat

## Installation
### Install pytorch according to your OS and the GPU/CPU option
Cf. https://pytorch.org/get-started/locally/
For example, with Windows, cuda 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

### Install bitsandbytes
For Windows:
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl

### Other packages
pip install -r requirements.txt

## Server configuration
Under .../chat_service/server/
   cp chat_server_config.xml.example chat_server_config.xml
   python chat_server.py