from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader

class LlamaFineTuneModel(L.LightningModule):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", learning_rate=1e-5 , tokenizer=None):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Load the pre-trained Llama model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        # ALTERAÇÃO 2: Usa o tokenizador fornecido, não cria um novo
        if tokenizer is None:
            raise ValueError("Um tokenizador deve ser fornecido ao modelo.")
        self.tokenizer = tokenizer

        # ALTERAÇÃO 3 (CRÍTICA): Redimensiona a camada de embedding do modelo
        # para corresponder ao vocabulário do tokenizador (que agora inclui o PAD).
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']  # <--- CORRIGIDO: Fornece os labels para o cálculo da perda
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids'] # <--- CORRIGIDO: Também na validação
        )
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def on_validation_epoch_end(self):
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #self.log('avg_val_loss', avg_loss)
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler with linear warmup and decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1  # 10% of training for warmup
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
