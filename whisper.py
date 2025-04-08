from datasets import load_dataset 
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import lightning 
import torch , pandas as pd 


""" 
Checklist 

baixar dataset mozzilla 

common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="validation")

fazer o dataloader, collate é necessário printar o tamanho dos áudios 

downsample para 16khz

montar a classe modelo que chama o whisper como modelo 

colocar o modelo large-v3 em 2 ou 3 gpus a100 h100, testar o tyni primeiro 

ver o feature_extractor , tokenizer , processor e model 

no final subir isso no huggingface 


from huggingface_hub import HfApi, HfFolder

print('Whisper small ')
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
api = HfApi()
folder = HfFolder()

folder.save_token(token)


# Upload training results to the Hub.
trainer.push_to_hub('Rafaelrosendo1/whisper_small_teste')
print('Trained model uploaded to the Hugging Face Hub')

"""