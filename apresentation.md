
### Introdução 


O que são LLM's?

Basicamente são modelos de deep learning que partilham de algumas caracteristias distintas, sejam elas:

 - NLP
 - Muitos dados
 - Transformers 
 - Aprendizado por reforço e não supervisionado
 - Camadas de atenção
 - Se propõe a ser um modelo multi-task


Modelos ASR e STT 

Temos modelo reconhecimento de fala e modelos que fazem tarefas como tradução e transcrição de texto a partir de áudio 

Esses modelos tendem a seguir um mesmo padrão como o processamento do áudio a partir do espectograma e a tokenização desses áudios e a alimentação dos modelos a partir dos dados gerados 

em suma eu vou apresentar dois modelos, no momento os dois do estado da arte que é o whisper e o seamlessm4t

### Whisper


Acredito que o que mais importa sobre o whisper não seja em si a sua estrutura enquanto modelo mas sim o que de features importantes o whisper pode oferecer, então aqui está o pipeline de processamento ao dar para o modelo um arquivo de áudio

### 1. **Entrada de Áudio**

O Whisper recebe um arquivo de áudio, tipicamente em um formato como `.wav`, como entrada. O arquivo de áudio contém a fala que precisa ser transcrita.

### 2. **Pré-processamento**

Antes que o áudio possa ser alimentado no modelo, ele passa por uma etapa de pré-processamento:

- **Amostragem**: O áudio é convertido para uma taxa de amostragem padronizada, se já não estiver no formato requerido.
- **Fracionamento**: O áudio é dividido em quadros ou segmentos de uma duração fixa (por exemplo, 25ms com uma sobreposição de 10ms).
- **Extração de Características**: Características acústicas são extraídas desses quadros. Características comuns incluem coeficientes cepstrais em frequência Mel (MFCCs), espectrogramas ou espectrogramas log-Mel, que representam o sinal de áudio de uma forma que a rede neural pode processar.

### 3. **Codificação**

As características de áudio pré-processadas são então passadas pela parte codificadora do modelo transformer:

- **Codificação Posicional**: Como a arquitetura transformer não entende inerentemente a ordem da sequência, codificações posicionais são adicionadas às características de entrada para fornecer ao modelo informações sobre a ordem dos quadros.
- **Mecanismo de Autoatenção**: O codificador usa mecanismos de autoatenção para ponderar a importância de diferentes quadros dentro da sequência de entrada, capturando informações contextuais em todo o segmento de áudio.
- **Camadas Empilhadas**: A entrada passa por várias camadas do codificador, com cada camada refinando ainda mais a representação das características de áudio.

### 4. **Decodificação**

Uma vez que a codificação está completa, o decodificador entra em ação para gerar as transcrições:

- **Geração de Tokens**: O decodificador começa a gerar tokens de texto (palavras ou subpalavras) um de cada vez. Em cada etapa, ele considera as características de áudio codificadas e os tokens gerados anteriormente.
- **Mecanismo de Atenção Cruzada**: O decodificador usa mecanismos de atenção cruzada para alinhar as características de áudio codificadas com o estado atual da geração de texto, garantindo que a transcrição corresponda com precisão às palavras faladas.
- **Busca em Feixe ou Decodificação Gulosamente**: Técnicas como busca em feixe podem ser usadas para gerar a sequência de tokens (palavras) mais provável, explorando múltiplos caminhos possíveis e selecionando o de maior probabilidade.

### 5. **Pós-processamento**

Os tokens de texto gerados são então convertidos em texto legível por humanos:

- **Des-tokenização**: A sequência de tokens é mesclada para formar palavras e frases coerentes.
- **Normalização de Texto**: O processamento adicional pode incluir restauração de pontuação, correção de maiúsculas e minúsculas, e formatação para produzir transcrições limpas e legíveis.


### SeamlessM4t 

Segue mais a onda do whisper como um encoder decoder transformer 

porém usa técnicas um pouco diferente como podem ver aqui na imagem 



### CTC transformers


Os CTC Transformers (Connectionist Temporal Classification Transformers) combinam o modelo de Transformer com a técnica de Connectionist Temporal Classification (CTC) para tarefas de reconhecimento de fala e transcrição de sequências de tempo, como reconhecimento de fala, OCR (Reconhecimento Óptico de Caracteres), e outras tarefas de sequência.

### Como Funcionam os CTC Transformers

1. **Entrada**:
    
    - A entrada pode ser uma sequência de vetores de características, como espectrogramas no caso de reconhecimento de fala ou imagens no caso de OCR.
2. **Processamento pelo Transformer**:
    
    - A entrada é processada pelas camadas de Transformer, que aplicam mecanismos de atenção para aprender dependências a longo prazo e padrões na sequência de entrada.
3. **Camada de Saída com CTC**:
    
    - A camada final do Transformer produz uma sequência de logits (saídas não normalizadas) que representam as probabilidades de cada símbolo ou caractere em cada posição da sequência.
    - A perda CTC é aplicada sobre esses logits para calcular a diferença entre a sequência predita e a sequência alvo, permitindo o treinamento do modelo sem a necessidade de alinhamentos explícitos.

### Vantagens dos CTC Transformers

- **Flexibilidade na Sequência de Saída**: Permite a produção de sequências de comprimento variável, o que é crucial para tarefas como reconhecimento de fala, onde a duração das palavras pode variar significativamente.
- **Eficiência**: O uso de Transformers permite o processamento paralelo das entradas, resultando em um treinamento mais rápido comparado a modelos sequenciais como LSTMs ou GRUs.
- **Capacidade de Generalização**: A combinação de Transformers com a perda CTC permite que o modelo capture dependências complexas na entrada, melhorando a precisão em tarefas de transcrição de sequência.

### Aplicações Comuns

- **Reconhecimento de Fala**: Transcrição de áudio em texto, onde o alinhamento exato entre o áudio e as palavras não é conhecido.
- **Reconhecimento Óptico de Caracteres (OCR)**: Transcrição de texto em imagens, como documentos digitalizados.
- **Transcrição de Música**: Converter partituras ou gravações de música em notação musical.


### Audio GPT 

1 Ao invés de treinar uma LLM do zero, foi usado uma LLM como chatGPT como motor para esse modelo AudioGPT, na implementação do AudioGPT foi usado modelos AST e TTS para serem conectados com a LLM via input e output 


### Etapas

4 etapas 

1 Modality transformation Usando input e output interface para modalização de transformação entre audio e texto, 

2 Análise de tarefa Utilizando o motor de dialogo e o prompt manager para ajudar o ChatGPT a entender a intenção do usuário para processar a informação de áudio 

3 Model assingment Recebendo as estruturas do argumento como "prosody" , timbre e controle de linguagem, o chatgpt chama o modelo de audio para entender e gerar 

4 Response Generation Geração e retornando uma resposta final para o usuário após a execução da Audio Foundation Models


Pode ser usado e tem repositório 


### Salmonn


Como audio encoder, Um transformador Q-Former é usado para concatenar os dois encoders em formato de audio tokens e por vez é imposto no modelo para sim gerar texto como mostra na figura

Salmonn usa o encoder do Whisper-Large-v2 como speech encoder e o BEATs como audio encoder e o Vicuna como LLM com 13B parametros como a espinha dorsal do modelo, para a window level Q-Former gerando janelas de aproximadamente 0.33 segundos por janela. gerando assim 88 textual tokens na saida do Q-Former para um áudio de 30 segundos 


### SpeechGPT

Para o Dataset foi usado em escala larga datasets de ASR em ingles, para tokenização foi usado o mHuBERT como o tokenizador ded speech para discretizar o speech data em unidades discretas e com isso remover unidades repetidas de fragmentos adjacentes para reduzir as unidades até obter 9 milhões de unidades 

task description 

Gerando ASR E TTS tarefas que são compativeis com speech-text data pairs, diferente the métodos de self instructc eles geraram descrições como zero-shot para ver como o modelo se sai 

estrutura do modelo
Discrete unit extractor = HuBERT 

LLM  = meta LLama e um teste com o GPT4


Unit Vocoder = HiFi-GAN 


Já foi negão só finalizar que é tua, se garantiu em kkkkkkkk