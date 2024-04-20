# COMP5423 Group Project

In this project, we construct a dialogue system based on the project datasets. The system implement the two options provided in the project requirements:

* A chatbot based on the BART model which is trained by our group.
* A chatbot based on a pre-trained large language model.

The system structure can be divided into three parts:

* A web interface to showcase our system.
* Two chatbots and a http server to provide the dialogue service.
* A BART model finetune framework to show how we train the model.

## Model Weight Files

According to the requirements, the model weight files belong to the large files, it is stored in the Google Drive. You should download the three files listed below and put them into the `/model` path in the source code directory.

* [model.safetensors](https://drive.google.com/file/d/16D97cC75pUWmQ0lhgcacx10rfMhTrfMh/view?usp=share_link)
* [generation_config.json](https://drive.google.com/file/d/16ft7nHkyIVpIAGqoxoIGrclxbyqQ7IKN/view?usp=share_link)
* [config.json](https://drive.google.com/file/d/14R4bkcJeicEbNxHWGGLouvQJfrpRpQkB/view?usp=share_link)


## Requirements

The required packages are listed in `requirements.txt`. You can find the file in the source code directory. Suppose you use [Anaconda](https://www.anaconda.com/) to manage the Python dependencies, you can install them by running:

```bash
conda create -n chatbot python=3.9.7
conda activate chatbot
pip install -r requirements.txt
```
## Quickstart

### Dialogue System

The Dialogue system consists of two parts: chatbot service and web interface.

1. The chatbot service is provided using http server.
   ```python
   python httpserver.py
   ```
2. Web Interface
   The system provide a static html web interface, find the html file under the path `/UI` in the source code directory. Open it using the browser (Chrome is highly recomended here).

### BART Model Fine-Tuning

```python
python finetune.py
```

Note that the output trained model weight files will be saved in the path `/model`.

## Evaluation



