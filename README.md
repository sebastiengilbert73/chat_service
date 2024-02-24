# chat_service
A service to chat

## Installation
### Install pytorch according to your OS and the GPU/CPU option
Cf. https://pytorch.org/get-started/locally/  
For example, with Windows, cuda 12.1:  
pip install torch --index-url https://download.pytorch.org/whl/cu121  

### Other packages
pip install -r requirements.txt

## Server configuration
Under .../chat_service/server/  
cp chat_server_config.xml.example chat_server_config.xml 
### Edit chat_server_config.xml to your needs. If you don't use a GPU, make sure to change the device to cpu
Note that without a GPU, the response may be very slow.

## Launch the server
python chat_server.py  

## Chainlit web interface
Under .../chat_service/chainlit_interface/  
cp app_config.xml.example app_config.xml  
chainlit run app.py -w  