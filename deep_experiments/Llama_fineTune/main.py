import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import LlamaFineTuneModel
from data import MyDataModule

def main(args):

    """
    Função principal para configurar e iniciar o treinamento do modelo.
    """
    # Define a semente para reprodutibilidade
    L.seed_everything(42, workers=True)

    # 1. Configuração do DataModule
    # Instancia o módulo de dados com os caminhos e hiperparâmetros
    data_module = MyDataModule(
        train_path=args.train_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_len=args.max_length,
        num_workers=args.num_workers
    )

    # 2. Configuração do Modelo
    # Instancia o modelo de fine-tuning
    model = LlamaFineTuneModel(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        tokenizer=data_module.tokenizer
    )

    # 3. Configuração dos Callbacks e Logger
    # Callback para salvar o melhor modelo com base na perda de validação
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Callback para parar o treinamento se a perda de validação não melhorar
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # Número de épocas sem melhora antes de parar
        verbose=True,
        mode='min'
    )
    
    # Logger para visualizar as métricas no TensorBoard
    tensorboard_logger = TensorBoardLogger("lightning_logs", name="llama_finetune")


    # 4. Configuração do Trainer
    # O Trainer do PyTorch Lightning gerencia todo o ciclo de treinamento
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        #strategy="fsdp",  
        max_epochs=args.max_epochs,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # Essencial para treinar LLMs grandes. Usa menos memória de GPU.
        precision='bf16-mixed',
        log_every_n_steps=10 # Loga a cada 10 batches
    )

    # No seu script principal (main.py), ANTES da chamada trainer.fit()

       # 5. Início do Treinamento
    print("Iniciando o treinamento...")
    trainer.fit(model, datamodule=data_module)
    print("Treinamento concluído.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Llama model on a text corpus.")

    # Argumentos relacionados ao ambiente e hardware
    parser.add_argument('--accelerator', type=str, default='gpu', help="Tipo de acelerador ('gpu', 'cpu').")
    parser.add_argument('--devices', type=int, default=1, help="Número de dispositivos a serem usados.")
    parser.add_argument('--num_workers', type=int, default=4, help="Número de workers para o DataLoader.")

    # Argumentos relacionados aos dados
    parser.add_argument('--train_path', type=str, required=True, help="Caminho para a pasta com os arquivos .txt de treino.")
    parser.add_argument('--max_length', type=int, default=1024, help="Comprimento máximo da sequência para o tokenizador.")

    # Argumentos relacionados ao modelo e treinamento
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help="Nome do modelo da Hugging Face.")
    parser.add_argument('--batch_size', type=int, default=1, help="Tamanho do lote (batch size). Use 1 se a memória for limitada.")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Taxa de aprendizado.")
    parser.add_argument('--max_epochs', type=int, default=3, help="Número máximo de épocas de treinamento.")
    
    args = parser.parse_args()
    
    # Verifica se o caminho do treino existe
    if not os.path.isdir(args.train_path):
        raise ValueError(f"O caminho especificado --train_path '{args.train_path}' não é um diretório válido.")
        
    main(args)
