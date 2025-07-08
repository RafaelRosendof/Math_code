
### Bibliotecas para importar #####
import torch
import torch.nn as nn , os 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class MNISTToGraph:
    """
    Classe responsável por converter imagens do MNIST em grafos.
    Cada pixel é um nó, e as arestas são baseadas na conectividade espacial (4 ou 8).
    Os nós têm características baseadas na intensidade do pixel e na posição normalizada.
    A classe também lida com imagens vazias, criando um grafo mínimo com um único nó.
    A conversão é feita para cada imagem individualmente, retornando um objeto Data do PyTorch Geometric.
    A classe pode ser configurada para usar conectividade 4 ou 8, dependendo do parâmetro `connectivity`.
    Esta classe é útil para preparar dados do MNIST para treinamento de redes neurais gráficas
    """
    
    def __init__(self, connectivity='8'):
        self.connectivity = connectivity
    
    def image_to_graph(self, image, label):
        
        # image shape: (28, 28)
        h, w = image.shape
        
        # Cria um no para cada pixel com intensidade normalizada
        # Normaliza a intensidade do pixel para o intervalo [0, 1]

        threshold = 0.1
        coords = np.where(image > threshold)
        
        if len(coords[0]) == 0:  # Se não houver pixels acima do limiar
            # Cria um grafo vazio com um único nó
            # Isso é necessário para evitar erros durante o treinamento
            x = torch.tensor([[0.0]], dtype=torch.float)
            edge_index = torch.tensor([[], []], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        
        # Node features: [pixel_intensity, normalized_x, normalized_y]
        node_features = []
        node_positions = {}
        
        for i, (row, col) in enumerate(zip(coords[0], coords[1])):
            intensity = image[row, col]
            norm_x = row / (h - 1)  # Normaliza a position
            norm_y = col / (w - 1)
            node_features.append([intensity, norm_x, norm_y])
            node_positions[(row, col)] = i
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Cria as arestas com base na conectividade
        # Usando conectividade 4 ou 8
        edges = []
        if self.connectivity == '4':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8-connectivity
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                         (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for (row, col), node_idx in node_positions.items():
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) in node_positions:
                    neighbor_idx = node_positions[(new_row, new_col)]
                    edges.append([node_idx, neighbor_idx])
        
        # Se não houver arestas, cria um grafo vazio
        if len(edges) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        
        ### No final é retornado um tensor Data do PyTorch Geometric
        # que contém as características dos nós, as arestas e o rótulo da imagem
        # x: características dos nós (intensidade do pixel, posição normalizada)
        # edge_index: arestas do grafo (conectividade entre os nós)
        # y: rótulo da imagem (dígito correspondente)
        return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

class MNIST_GNN(nn.Module):
    """
    A classe MNIST_GNN define uma rede neural gráfica (GNN) para classificação de dígitos do MNIST.
    Ela utiliza camadas de convolução GCN (Graph Convolutional Network) para extrair características dos grafos representando imagens do MNIST.
    A arquitetura consiste em várias camadas GCN seguidas por normalização em lote (Batch Normalization) e pooling global.
    A saída é processada por uma camada totalmente conectada para prever a classe do dígito.
    A classe é configurável com parâmetros como dimensão de entrada, dimensão oculta, número de classes e número de camadas.
    A função forward define o fluxo de dados através da rede, aplicando convoluções, normalizações, pooling e finalmente a classificação.
    A saída é uma distribuição de probabilidade logarítmica sobre as classes, adequada para treinamento com perda de entropia cruzada negativa (NLLLoss).
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=10, num_layers=3):
        super(MNIST_GNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Primeira camada
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Camadas intermediárias
        # Adiciona camadas GCN e BatchNorm para as camadas intermediárias
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Última camada
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Camada de classificação
        # A camada de classificação combina as saídas de pooling global (média e máxima)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Aplicação das camadas GCN e BatchNorm
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Pooling global
        # Calcula pooling global médio e máximo
        # Isso combina as características dos nós em uma representação fixa para a classificação
        # global_mean_pool calcula a média das características dos nós por grafo
        # global_max_pool calcula o máximo das características dos nós por grafo
        # As duas representações são concatenadas para formar a entrada da camada de classificação
        # batch contém o índice do grafo ao qual cada nó pertence
        # Isso é necessário para agrupar as características dos nós corretamente
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Saida da rede
        # A saída é passada pela camada de classificação, que produz logits para cada classe
        # A função log_softmax é aplicada para obter a distribuição de probabilidade logarítmica
        # Isso é adequado para treinamento com perda de entropia cruzada negativa (NLLLoss)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def create_mnist_graph_dataset(train=True, connectivity='8'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST dataset
    # Usando funções built-in do PyTorch para baixar o dataset MNIST
    # O parâmetro `train` determina se é o conjunto de treinamento ou teste
    # O parâmetro `connectivity` define a conectividade dos grafos (4 ou 8)
    # A normalização é aplicada para ajustar os valores dos pixels
    # A normalização é feita com base na média e desvio padrão do MNIST
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    converter = MNISTToGraph(connectivity=connectivity)
    
    graph_data = []
    print(f"Converting {'train' if train else 'test'} set to graphs...")
    
    for i, (image, label) in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} images")
        
        image_np = image.squeeze().numpy()
        graph = converter.image_to_graph(image_np, label)
        graph_data.append(graph)
    
    return graph_data

def train_model(model, train_loader, optimizer, device):
    
    """
    Função para treinar o modelo MNIST_GNN.
    Ela itera sobre os lotes de dados do DataLoader, calcula a perda usando Loss,
    realiza a retropropagação e atualiza os pesos do modelo.
    A função retorna a perda média e a acurácia do treinamento.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(train_loader), correct / total

def test_model(model, test_loader, device):
    """
    Função para testar o modelo MNIST_GNN.
    Ela avalia o modelo em um conjunto de dados de teste, calcula a acurácia
    e retorna a acurácia média.
    O modelo é colocado em modo de avaliação (eval) para desativar dropout e batch normalization.
    A função itera sobre os lotes de dados do DataLoader, faz previsões,
    e compara as previsões com os rótulos verdadeiros para calcular a acurácia
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
    
    return correct / total

def main():
    
    # Função principal para treinar o modelo MNIST_GNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    print("Creating graph datasets...")
    train_graphs = create_mnist_graph_dataset(train=True)  
    test_graphs = create_mnist_graph_dataset(train=False)
    
    # Criando os dataloaders 
    
    # DataLoader é usado para carregar os dados em lotes durante o treinamento e teste
    # Ele embaralha os dados de treinamento para melhorar a generalização do modelo
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Inicializando o modelo com o objeto model 
    model = MNIST_GNN(input_dim=3, hidden_dim=64, num_classes=10, num_layers=3)
    model = model.to(device) #carregando o modelo para o dispositivo (GPU ou CPU)
    
    # Definindo o otimizador e o scheduler
    # O otimizador Adam é usado para atualizar os pesos do modelo
    # O scheduler reduz a taxa de aprendizado a cada 10 épocas
    # Isso ajuda a estabilizar o treinamento e melhorar a convergência
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Loop de treinamento
    # O loop de treinamento itera por N épocas, treinando o modelo em cada época
    # Ele calcula a perda média e a acurácia do treinamento, além de avaliar o modelo no conjunto de teste
    # As perdas e acurácias são armazenadas para plotagem posterior
    # O modelo é salvo após o treinamento para uso futuro
    print("Starting training...")
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(200):
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        test_acc = test_model(model, test_loader, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1:02d}: Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    
    # Salvando o modelo treinado
    # O modelo é salvo no formato .pth para uso futuro
    torch.save(model.state_dict(), 'mnist_gnn.pth')
    
    # Plotando as perdas e acurácias
    # As perdas de treinamento e as acurácias são plotadas para visualizar o desempenho
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final test accuracy: {test_accs[-1]:.4f}")



def load_model(model_path, device='cpu'):
    """Load a trained MNIST GNN model"""
    model = MNIST_GNN(input_dim=3, hidden_dim=64, num_classes=10, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(photo_path):
    """Preprocess an image for MNIST prediction"""
    # Load image
    if isinstance(photo_path, str):
        image = Image.open(photo_path)
    else:
        image = photo_path
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply MNIST normalization (mean=0.1307, std=0.3081)
    image_array = (image_array - 0.1307) / 0.3081
    
    return image_array

def inferencePhoto(photo_path, model_path, device='cpu', show_image=True):
    """
    Predict digit from photo using trained MNIST GNN model
    
    Args:
        photo_path (str): Path to the image file
        model_path (str): Path to the saved model (.pth file)
        device (str): Device to run inference on ('cpu' or 'cuda')
        show_image (bool): Whether to display the input image
    
    Returns:
        dict: Prediction results with confidence scores
    """
    
    # Check if files exist
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Image file not found: {photo_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Preprocess image
    print("Preprocessing image...")
    image_array = preprocess_image(photo_path)
    
    # Convert image to graph
    converter = MNISTToGraph(connectivity='8')
    # Use dummy label (0) since we're doing inference
    graph_data = converter.image_to_graph(image_array, 0)
    
    # Create a batch with single graph
    graph_data = graph_data.to(device)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        output = model(graph_data)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get all class probabilities
    class_probs = {i: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nAll class probabilities:")
    for digit, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  Digit {digit}: {prob:.4f}")
    
    # Show image if requested
    if show_image:
        plt.figure(figsize=(8, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        original_img = Image.open(photo_path)
        if original_img.mode != 'L':
            original_img = original_img.convert('L')
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed image (28x28)
        plt.subplot(1, 2, 2)
        # Denormalize for display
        display_img = (image_array * 0.3081) + 0.1307
        display_img = np.clip(display_img, 0, 1)
        plt.imshow(display_img, cmap='gray')
        plt.title(f'Processed (28x28)\nPredicted: {predicted_class} ({confidence:.3f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'predicted_digit': predicted_class,
        'confidence': confidence,
        'all_probabilities': class_probs
    }

def test_inference_example():
    """Test inference with a sample from MNIST test set"""
    # Download o dataset de teste 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Salvando um exemplo de imagem do dataset de teste
    # Isso é útil para testar a função de inferência com uma imagem real
    sample_image, true_label = test_dataset[0]
    sample_image_pil = transforms.ToPILImage()(sample_image)
    sample_image_pil.save('test_digit.png')
    
    #print(f"Saved test image 'test_digit.png' with true label: {true_label}")
    #print("You can now test inference with: inferencePhoto('test_digit.png', 'mnist_gnn.pth')")

if __name__ == "__main__":
    # treina e salva o modelo 
    model = main()
    

    test_inference_example()

