import re 
import chardet
import os 
import lightning as L
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


# Existe arquivos de dados em formato csv, pdf , txt e vou ver se tem mais, 
#### FAZER UM NOTEBOOK PARA EXPLORAR OS DADOS #### 

### Feito 
class CustomDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=2048):
        # Corrigido um pequeno erro de digitação de 'file_pahts' para 'file_paths'
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)
    
    # abre, lê e processa um único arquivo de texto
    '''
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text = clean_text(text)

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # --- ALTERAÇÃO PRINCIPAL ---
        # 1. Adicionar os labels, que são uma cópia dos input_ids.
        #    Usamos .clone() para garantir que seja uma cópia independente.
        inputs['labels'] = inputs['input_ids'].clone()

        # 2. Remover a linha .squeeze(0). O DataLoader lida com isso.
        #    Isso retorna um dicionário com tensores no formato [1, max_length].
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        
        return inputs
    '''

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        text = ""
        try:
            # 1. Abre o arquivo em modo binário ('rb') para ler os bytes
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                # 2. Usa o chardet para detectar a codificação
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # 3. Decodifica o texto usando a codificação detectada
            # Adiciona um fallback para 'utf-8' caso a detecção falhe
            if encoding:
                text = raw_data.decode(encoding)
            else:
                print(f"Aviso: Não foi possível detectar a codificação para {file_path}. Usando UTF-8 como padrão.")
                text = raw_data.decode('utf-8', errors='ignore') # 'ignore' remove caracteres problemáticos

        except Exception as e:
            # Captura qualquer outro erro que possa ocorrer na leitura
            print(f"Erro ao ler o arquivo {file_path}: {e}")
            text = "" # Retorna texto vazio se houver erro
                
        text = clean_text(text)
        
        # O resto do seu código de tokenização continua igual...
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
        inputs['labels'] = inputs['input_ids'].clone()
        
        # IMPORTANTE: Mantenha esta linha para remover a dimensão extra
        final_inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        
        return final_inputs
       # return inputs


class MyDataModule(L.LightningDataModule):
    def __init__(self, train_path, model_name="meta-llama/Llama-3.2-1B", batch_size=4, max_len=2048, num_workers=4):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Corrigido o erro de digitação aqui
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Adiciona um token de padding se o modelo não tiver um. Llama não tem.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # ja tenho uma pasta com os arquivos txt, então não preciso baixar nada, apenas ler os arquivos
    def prepare_data(self):
        pass 

    def setup(self , stage=None):
        # Ler os arquivos txt da pasta train_path
        all_files = [os.path.join(self.hparams.train_path , f) for f in os.listdir(self.hparams.train_path) if f.endswith('.txt')]
        train_paths, val_paths = train_test_split(all_files, test_size=0.1, random_state=42)

        self.train_dataset = CustomDataset(
            train_paths,
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_len
        )

        self.val_dataset = CustomDataset(
            val_paths,
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_len
        )

        print(f"Dados carregados: {len(self.train_dataset)} exemplos de treino, {len(self.val_dataset)} exemplos de validação.")
         

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True # Otimização para carregar dados para a GPU
        )
    
    

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    # Não tenho teste, então não preciso implementar esse método
    def test_dataloader(self):
        pass 
