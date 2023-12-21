'''
Modelo de aprendizado de máquina para prever o preço de uma residência em Melbourne utilizando Decision Tree Regressor

Script baseado na lição 3 do curso "Intro to Machine Learning" do Kaggle (https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model)
Adaptado por Arthur Castro - Novembro de 2023

Observação: Você pode descomentar as linhas de 'print()' que desejar para visualizar algumas etapas intermediárias do código
'''

# ========== Importando Bibliotecas ==========
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


print('==================== Etapa 1 - Selecionando os Dados ====================\n\n')

# Junto a este código você encontrará o arquivo 'melb_data.csv', que consiste em dados de diversas residências de Melbourne.
# Se for necessário, modifique a linha abaixo para que a variável 'melbourne_file_path' indique corretamente o caminho do arquivo csv em seu ambiente.
melbourne_file_path = './curso_kaggle_intro_to_ml/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

print(f"Visualizando o dataframe que nomeamos de 'melbourne_data':\n{melbourne_data}\n\n")
print(f"Visualizando as colunas do dataframe 'melbourne_data':\n{melbourne_data.columns}\n\n")

'''
    Observação:
    
    A base de dados que estamos utilizando apresenta alguns campos faltantes. Isto significa que determinadas casas cadastradas não tem o registro de algumas de suas características.
    Para que isto não atrapalhe o treinamento do nosso modelo, a lição original do kaggle orienta que se excluam todas as linhas que possuam algum valor NaN executando:
    
    melbourne_data = melbourne_data.dropna(axis=0)  # Excluindo todas as linhas de 'melbourne_data' que contenham ao menos um valor NaN

    No entanto, foi percebido que os valores faltantes encontram-se exatamente em colunas que já não pretendemos utilizar no código atual, ou seja, para as 'features' escolhidas não se faz necessária a execução de 'dropna'.
    Caso você pretenda utilizar diferentes 'features' lembre-se de avaliar os campos que contém NaN. Você pode tratar seu dataframe com 'dropna(axis=0)' para excluir as linhas ou 'dropna(axis=1)' para excluir as colunas.
'''

# Selecionando a 'Prediction Target' (o que se quer prever, a 'saída' do modelo):
y = melbourne_data.Price
print(f"Visualizando 'y':\n{y}\n\n")

# Escolhendo as 'Features' (as 'entradas' do modelo):
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(f"Visualizando 'X':\n{X}\n\n")
print(f"Visualizando estatísticas dos dados de 'X':\n{X.describe()}\n\n")
print(f"Visualizando os primeiros 5 registros de 'X':\n{X.head()}\n\n")


print('==================== Etapa 2 - Construindo o Modelo ====================\n\n')

# Definindo o modelo:
melbourne_model = DecisionTreeRegressor()

# Treinando o modelo:
melbourne_model.fit(X, y)


print('==================== Etapa 3 - Executando Previsões ====================\n\n')

previsao = melbourne_model.predict(X.head())
print(f"Visualizando os valores previstos para as primeiras 5 casas registradas:\n {previsao}\n\n")


print('==================== Etapa 4 - Validando o Modelo ====================\n\n')

print(f"Consultando os valores registrados das 5 primeiras casas no dataframe:\n {y.head()}\n\n")

print(f"Agora é possível comparar os preços previstos (na etapa 3) com os preços reais cadastrados (na etapa 4) e avaliar o modelo!\n")
