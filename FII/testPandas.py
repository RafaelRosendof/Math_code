import pandas as pd

# 1. Definir a URL da tabela de resultados de FIIs do Fundamentus
url = 'https://www.fundamentus.com.br/fii_resultado.php'

# 2. O Pandas vai ler a página e identificar a tabela automaticamente
# O header=0 diz que a primeira linha são os títulos das colunas
# O decimal=',' e thousands='.' ajudam a converter o formato brasileiro de números
dados = pd.read_html(url, decimal=',', thousands='.')[0]

# 3. Limpeza básica (O site às vezes manda % como texto, precisamos converter)
# Vamos remover o símbolo '%' e converter para número nas colunas importantes
cols_porcentagem = ['Dividend Yield', 'Vacância Média']

for col in cols_porcentagem:
    # Transforma texto em string, remove o %, troca vírgula por ponto e vira float
    dados[col] = dados[col].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)

# 4. APLICANDO SUA ESTRATÉGIA (Exemplo)
# Filtros:
# - Liquidez diária > 500 mil reais (para conseguir vender fácil)
# - P/VP entre 0.8 e 1.1 (nem muito barato suspeito, nem muito caro)
# - Dividend Yield > 0 (tem que estar pagando)

meus_fiis = dados[
    (dados['Liquidez'] > 500000) &
    (dados['P/VP'] > 0.8) &
    (dados['P/VP'] < 1.1) &
    (dados['Dividend Yield'] > 0)
]

# 5. Ordenar pelos que pagam mais (DY)
ranking = meus_fiis.sort_values(by='Dividend Yield', ascending=False)

# Mostrar os Top 10
print(ranking[['Papel', 'Segmento', 'Cotação', 'Dividend Yield', 'P/VP']].head(10))