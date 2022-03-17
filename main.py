from pandas_datareader import data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os

cotacaoBovespa = web.DataReader('^BVSP', data_source='yahoo', start='06-01-2020', end='06-01-2021')

print(cotacaoBovespa.head())

#cotacaoBovespa['Adj Close'].plot(figsize=(20,10))
#plt.show()

cotacaoPetrobras = web.DataReader('PETR4.SA', data_source='yahoo', start='06-01-2020', end='06-01-2021')

print(cotacaoBovespa.head())

#cotacaoBovespa['Adj Close'].plot(figsize=(20,10))
#plt.show()


# Buscado varias empresas

# empresas = ['PETR4.SA', 'MGLU3.SA', 'MGLU3.SA', '^BVSP']

# for empresa in empresas: print(empresa) cotacao = web.DataReader(f'{empresa}', data_source='yahoo', start='06-01-2020', end='06-01-2021') cotacao['Adj Close'].plot(figsize=(20,10)) plt.show()

# Empresas que fazem parte do indice bovesta
# #url = 'https://www.onze.com.br/blog/acoes-ibovespa/'

# #df = pd.read_html(url)

# #empresas = list(df[0][0].copy())

# #print(empresas)


# É preciso fazer o download de todas os balanços de cada empresa, coloquei todos na pasta balancos

# Aramazenaremos todas os balanços e dres das empresas em um dicionario 'fundamentos'

fundamentos = {}

arquivos = os.listdir('balancos')

empresas = [ str(arquivo.split('_')[1].split('.')[0]) for arquivo in arquivos ]

print(len(empresas))

empresas = ["ABEV3", "AZUL4", "BTOW3", "B3SA3", "BBSE3", "BRML3", "BBDC4", "BRAP4", "BBAS3", "BRKM5", "BRFS3", "BPAC11", "CRFB3", "CCRO3", "CMIG4", "HGTX3", "CIEL3", "COGN3", "CPLE6", "CSAN3", "CPFE3", "CVCB3", "CYRE3", "ECOR3", "ELET6", "EMBR3", "ENBR3", "ENGI11", "ENEV3", "EGIE3", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GOAU4", "GOLL4", "NTCO3", "HAPV3", "HYPE3", "IGTA3", "GNDI3", "ITSA4", "ITUB4", "JBSS3", "JHSF3", "KLBN11", "RENT3", "LCAM3", "LAME4", "LREN3", "MGLU3", "MRFG3", "BEEF3", "MRVE3", "MULT3", "PCAR3", "PETR4", "BRDT3", "PRIO3", "QUAL3", "RADL3", "RAIL3", "SBSP3", "SANB11", "CSNA3", "SULA11", "SUZB3", "TAEE11", "VIVT3", "TIMS3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VVAR3", "WEGE3", "YDUQ3"]


def ajustar (df,nome):
    df.iloc[0, 0] = nome
    df.columns = df.iloc[0]
    df = df[1:]
    df.set_index(nome)
    return df
    
for arquivo in arquivos:
    
    nome = str(arquivo.split('_')[1].split('.')[0])
    
    if nome in empresas:
        
        balanco = ajustar(pd.read_excel(f'balancos/{arquivo}', sheet_name=0),nome)
        dre = ajustar(pd.read_excel(f'balancos/{arquivo}', sheet_name=1),nome)
        
        fundamentos[nome] = balanco.append(dre)

print(len(fundamentos))

#Armazenando cotações em um dicionario 'cotacoes'

cotacoes_df = pd.read_excel('Cotacoes.xlsx')

cotacoes = {}

for empresa in cotacoes_df['Empresa'].unique():
    cotacoes[empresa] = cotacoes_df.loc[cotacoes_df['Empresa'] == empresa, :]
print(cotacoes['ABEV3'].head())
print(len(cotacoes))


#removendo empresas que não contem todo o periodo de ações realizado
for empresa in empresas:
    if cotacoes[empresa].isnull().values.any():
        cotacoes.pop(empresa)
        fundamentos.pop(empresa)
        
print(len(cotacoes), len(fundamentos))



# Mescando os dois dicts
print(fundamentos["ABEV3"])

for empresa in fundamentos:
        
    tabela = fundamentos[empresa].T
    
    if not tabela.index.name:
        tabela.columns = tabela.iloc[0]
        tabela = tabela.iloc[1:]
        
    tabela.index = pd.to_datetime(tabela.index, format='%d/%m/%Y')
    
    tabela_cotacao = cotacoes[empresa].set_index('Date')
    tabela_cotacao = tabela_cotacao[['Adj Close']]
    
    tabela = tabela.merge(tabela_cotacao, right_index=True, left_index=True)
    
    tabela.index.name = empresa
    
    fundamentos[empresa] = tabela

print(len(fundamentos))
print(fundamentos['ABEV3'].head())



# Remover empresas que trabalha com tipos dados diferentes, comparando as colunas
colunas = list(fundamentos['ABEV3'].columns)

fund = fundamentos.copy()

for empresa in fundamentos:
    if set(colunas) != set(fundamentos[empresa].columns):
        fund.pop(empresa)
        
fundamentos = fund.copy()

print(len(fundamentos))



#alterando nomes de colunas repetidos

texto_colunas = ";".join(colunas)

colunas_modificadas = []

for coluna in colunas:
    
    if colunas.count(coluna) > 1 and coluna not in colunas_modificadas:
        
        texto_colunas = texto_colunas.replace(';' + coluna + ';', ';' + coluna + '_1;', 1)
        
        colunas_modificadas.append(coluna)
        
colunas = texto_colunas.split(';')


for empresa in fundamentos:
    
    fundamentos[empresa].columns = colunas


# Remover colunas com valores nulos
valores_vazios = dict.fromkeys(colunas, 0)

total_linhas = 0

for empresa in fundamentos:
    
    tabela = fundamentos[empresa]
    
    total_linhas += tabela.shape[0]
    
    for coluna in colunas:
        
        qtde_vazios = pd.isnull(tabela[coluna]).sum()
        
        valores_vazios[coluna] += qtde_vazios
        
print(valores_vazios)
print('\n', total_linhas)


remover_colunas = [ coluna for coluna in valores_vazios if valores_vazios[coluna] > 50 ]

for empresa in fundamentos:
    fundamentos[empresa] = fundamentos[empresa].drop(remover_colunas, axis=1)
    fundamentos[empresa] = fundamentos[empresa].ffill()  #substitui os valores vazios com os mesmos valores da linha a cima
print(fundamentos['ABEV3'].shape)



# Analise de regras para IA
# Vamos criar tres regras, comprar, não comprar e vender (se o modelo analisar apenas quais comprar a probabilidade de erros é maior)...

# Subio mais do que o bovespa (ou caiu menos) -> COMPRAR (valor = 2)
# Subio menos que o bovespa até 2% (ou caiu mais que bovespa até 2%) -> NÃO COMPRAR (valor = 1)
# Subio menos que o bovespa mais que 2% (ou caiu mais que o bovespa mais que 2%) -> VENDER (valor = 0)
# Para isso analiseramenos as porcentagens de variações entre os trimestres

data_inical = '12/20/2012'
data_final = '04/20/2021'

df_BVSP = web.DataReader('^BVSP', data_source='yahoo', start = data_inical, end=data_final)

print(df_BVSP.head(3))



datas = fundamentos['ABEV3'].index

print(datas)


for data in datas:
    if not data in df_BVSP.index:
        df_BVSP.loc[data] = np.nan
        
df_BVSP = df_BVSP.sort_index()
df_BVSP = df_BVSP.ffill()
df_BVSP = df_BVSP.rename(columns={'Adj Close': 'IBOV'})

#display(df_BVSP)


for empresa in fundamentos:
    fundamentos[empresa] = fundamentos[empresa].merge(df_BVSP[['IBOV']], left_index=True, right_index=True)
    
print(fundamentos['ABEV3'].head(3))

# Tranformar indicadores em percentuais
# % trimestre de fundamentos = fundamentos trimestre atual / fundamentos do trimestre anterior (-1)

# % trimestre da cotação = cotação trimestre seguinte (+1) / cotação do trimestre atual



for empresa in fundamentos:
    
    fundamento = fundamentos[empresa].sort_index()
    
    for coluna in fundamento:
         
        if 'Adj Close' not in coluna and 'IBOV' not in coluna:
        
            condicoes = [
                (fundamento[coluna].shift(1) > 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] > 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) == 0) & (fundamento[coluna] > 0),
                (fundamento[coluna].shift(1) == 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] == 0),
            ]
            valores = [
                -1,
                1,
                (abs(fundamento[coluna].shift(1))-abs(fundamento[coluna])) / abs(fundamento[coluna].shift(1)),
                1,
                -1,
                1,
            ]
            
            fundamento[coluna] = np.select(condicoes, valores, default= fundamento[coluna] / fundamento[coluna].shift(1) - 1)
            
            
    fundamento['Adj Close'] = fundamento['Adj Close'].shift(-1) / fundamento['Adj Close'] -1
    fundamento['IBOV'] = fundamento['IBOV'].shift(-1) / fundamento['IBOV'] -1
            
    fundamento['Resultado'] = fundamento['Adj Close'] - fundamento['IBOV']
            
    condicoes = [
            (fundamento['Resultado'] > 0),
            (fundamento['Resultado'] < 0) & (fundamento['Resultado'] >= -0.02),
            (fundamento['Resultado'] < -0.02),
        ]
    valores = [2,1,0]
            
    fundamento['Decisao'] = np.select(condicoes, valores) 
    
    fundamentos[empresa] = fundamento
print(fundamentos['ABEV3'].head(3))



# Tratando dados
colunas = list(fundamentos['ABEV3'].columns)

valores_vazios = dict.fromkeys(colunas, 0)

total_linhas = 0

for empresa in fundamentos:
    
    tabela = fundamentos[empresa]
    
    total_linhas += tabela.shape[0]
    
    for coluna in colunas:
        
        qtde_vazios = pd.isnull(tabela[coluna]).sum()
        
        valores_vazios[coluna] += qtde_vazios
        

remover_colunas = [ coluna for coluna in valores_vazios if valores_vazios[coluna] > total_linhas / 3 ]

for empresa in fundamentos:
    fundamentos[empresa] = fundamentos[empresa].drop(remover_colunas, axis=1)
    fundamentos[empresa] = fundamentos[empresa].fillna(0)  #substitui os valores vazios por 0

print(fundamentos['ABEV3'].head(3))



for empresa in fundamentos:
    
    fundamentos[empresa] = fundamentos[empresa].drop(['Adj Close', 'IBOV', 'Resultado'], axis=1)



# Compilando Todos DF em um unico DF
copia_fundamentos = fundamentos.copy()
base_dados = pd.DataFrame()

for empresa in copia_fundamentos:
    
    copia_fundamentos[empresa] = copia_fundamentos[empresa][1:-1]
    copia_fundamentos[empresa] = copia_fundamentos[empresa].reset_index(drop=True)
    
    base_dados = base_dados.append(copia_fundamentos[empresa])
print(base_dados.shape)
# Analise exploratória
# Visualização dos dados - qtde de dicições de cada tipo
base_dados['Decisao'].value_counts(normalize=True).map('{:.1%}'.format)
grafico = px.histogram(base_dados, x='Decisao', color='Decisao')
grafico.show()



# Remover valores de decisao 1 = não comprar => 0 = vender

base_dados.loc[base_dados['Decisao']==1, 'Decisao'] = 0

base_dados['Decisao'].value_counts(normalize=True).map('{:.1%}'.format)



# Analise de correlação (seaborn)
# Criando tabela de correlações com pandas

correlacoes = base_dados.corr()
# grafico seaborn

fig, ax = plt.subplots(figsize=(25,20))

sns.heatmap(correlacoes, ax=ax, cmap='Wistia')

plt.show()


# Remover correlações maiores 0.8 (não atrapalhar decisões da IA)

def identificarCorrelacoesEntreColunas(df):

    correlacoes_encontradas = []

    for coluna in df:
        for linha in df.index:

            if linha != coluna:

                valor = abs(df.loc[linha,coluna])

                if valor > 0.8 and (coluna, linha, valor) not in correlacoes_encontradas:

                    correlacoes_encontradas.append((linha, coluna, valor))

    remover_colunas = [ tup[0] for tup in correlacoes_encontradas if tup[1] == 'Ativo Total']

    return list(remover_colunas)

remover_colunas = identificarCorrelacoesEntreColunas(correlacoes)

print(len(remover_colunas))


base_dados = base_dados.drop(remover_colunas, axis=1)


correlacoes = base_dados.corr()

remover_colunas = identificarCorrelacoesEntreColunas(correlacoes)

print(len(remover_colunas))

fig, ax = plt.subplots(figsize=(25,20))

sns.heatmap(correlacoes, ax=ax, cmap='Wistia')

plt.show()




# Feature selection
# Treinar uma inteligencia artificial para apontar quais colunas tem maior relevancia e reduzir a quantidade de colunas que serão estudadas pela IA
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier(random_state=1)

x = base_dados.drop('Decisao', axis=1)
y = base_dados['Decisao']

modelo.fit(x,y)

caracteristicas_importantes = pd.DataFrame(modelo.feature_importances_, x.columns).sort_values(by=0, ascending=False)

top10 = list(caracteristicas_importantes.index)[:10]

print(top10)



# Standard Scaler - padorniza a base de dados para melhorar a IA
from sklearn.preprocessing import StandardScaler
def ajustar_scaler (df, campo_respostas):
    
    scaler = StandardScaler()
    
    df_auxiliar = df.drop(campo_respostas, axis=1)
    
    df_auxiliar = pd.DataFrame(scaler.fit_transform(df_auxiliar), df_auxiliar.index, df_auxiliar.columns)
    
    df_auxiliar[campo_respostas] = df[campo_respostas]
    
    return df_auxiliar
nova_base_dados = ajustar_scaler(base_dados, 'Decisao')
top10.append('Decisao')
nova_base_dados = nova_base_dados[top10].reset_index(drop=True)

display(nova_base_dados.head(3))





# Criar comparativo e avaliação
# Separar Dados de Treino e Testes
from sklearn.model_selection import train_test_split
x = nova_base_dados.drop('Decisao', axis=1)
y = nova_base_dados['Decisao']
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, random_state=1)
# Dummy Classifier
# Chuta os resultados na mesma proporção que a base (assim comparamos se a IA é molhor do que o aleatório)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
dummy = DummyClassifier(strategy='stratified', random_state=2)

dummy.fit(x_treino, y_treino)
previsao_dummy = dummy.predict(x_teste)
# Metricas de Avaliação
# precision é a metrica principal
# recall também é útil mas precision é mais importante!
def avaliar (y_teste, previsao, nome_modelo):
    
    print(f'MODELO: {nome_modelo}','\n')
    report = classification_report(y_teste, previsao)
    cf_matrix = pd.DataFrame(confusion_matrix(y_teste, previsao), index=['Vender', 'Comprar'], columns=['Vender', 'Comprar'])
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt=',')
    print(report)
    print('\n','Colunas são decisões do modelo, linhas quais eram respostas certas')
    plt.show()
    print('-'*120)
avaliar(y_teste, previsao_dummy, "Dummy")







# - Treinar modelos de IA
# Modelos de IA que vamos testar
# AdaBoost
# Decision Tree
# Random Forest
# ExtraTree
# Gradient Boost
# K Nearest Neighbors (KNN)
# Logistic Regression
# Naive Bayes
# Support Vector Machine (SVM)
# Rede Neural
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

modelos = {
    "AdaBoost": AdaBoostClassifier(random_state=1),
    "DecisionTree": DecisionTreeClassifier(random_state=1),
    "RandomForest": RandomForestClassifier(random_state=1),
    "ExtraTree": ExtraTreesClassifier(random_state=1),
    "GradientBoost": GradientBoostingClassifier(random_state=1),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(random_state=1),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(random_state=1),
    "RedeNeural": MLPClassifier(random_state=1, max_iter=400),
}
# Treinar todos os modelos importados
for nome_modelo in modelos:
    modelo = modelos[nome_modelo]
    modelo.fit(x_treino, y_treino)
    previsoes = modelo.predict(x_teste)
    
    avaliar(y_teste, previsoes, nome_modelo)
    
    modelos[nome_modelo] = modelo




# Melhorar o melhor modelo
# modelo escolhido: RandomForest
# Metricas de decisões de: 'COMPRAR'
# Criterio principal:
# precision: 0.56
# Criterio de desempate:
# recall: 0.53
# Tunning do modelo
# Parametros padrões do modelo
# n_estimators = 100
# max_features = 'auto'
# min_sample_split = 2
# gridsearch -> testa as combinações

modelo_final = modelos["RandomForest"]

n_estimators = range(10, 251, 30)
max_features = list(range(2,11,2))
max_features.append('auto')
min_samples_split = range(2,11,2)
#entre os valores testados garamtimos que os valores padrões serão inclusos
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score
precision2_score = make_scorer(precision_score, labels=[2], average='macro')
grid = GridSearchCV(
    estimator= RandomForestClassifier(),
    param_grid={
        'n_estimators': n_estimators,
        'max_features': max_features,
        'min_samples_split': min_samples_split,
        'random_state': [1],
    },
    scoring=precision2_score
)

resultado_grid = grid.fit(x_treino, y_treino)
print('Treino concluido!')
modelo_melhorado = resultado_grid.best_estimator_
previsoes = modelo_melhorado.predict(x_teste)

avaliar(y_teste, previsoes, 'RandomForest Melhorado')



# Aplicar na pratica e testar resultados
ultimo_trimestre_fundamentos = fundamentos.copy()

ultimo_trimestre_base_dados = pd.DataFrame()

lista_empresas = []
for empresa in ultimo_trimestre_fundamentos:
    
    ultimo_trimestre_fundamentos[empresa] = ultimo_trimestre_fundamentos[empresa][-1:]
    ultimo_trimestre_fundamentos[empresa] = ultimo_trimestre_fundamentos[empresa].reset_index(drop=True)
    lista_empresas.append(empresa)
    
    ultimo_trimestre_base_dados = ultimo_trimestre_base_dados.append(ultimo_trimestre_fundamentos[empresa])
    
print(ultimo_trimestre_base_dados)
print(lista_empresas)




ultimo_trimestre_base_dados = ultimo_trimestre_base_dados.reset_index(drop=True)
ultimo_trimestre_base_dados = ultimo_trimestre_base_dados[top10]
ultimo_trimestre_base_dados = ajustar_scaler(ultimo_trimestre_base_dados, 'Decisao')
ultimo_trimestre_base_dados = ultimo_trimestre_base_dados.drop('Decisao', axis=1)
# Comparando decisões da inteligencia artifical e analisando redimento

previsoes = modelo_melhorado.predict(ultimo_trimestre_base_dados)
print(previsoes)



carteira = []
carteira_inicial = []
investir_empresas = []

for i, empresa in enumerate(lista_empresas):
    if previsoes[i] == 2:
        
        investir_empresas.append(empresa)
        
        #supondo que investiria 1000 reais em cada empresa
        carteira_inicial.append(1000)
        
        cotacao = cotacoes[empresa]
        cotacao = cotacao.set_index('Date')
        
        cotacao_inicial = cotacao.loc['2020-12-31', 'Adj Close']
        cotacao_final = cotacao.loc['2021-03-31', 'Adj Close']
        
        precentual = cotacao_final / cotacao_inicial
        
        carteira.append(1000 * precentual)

saldo_inicial = sum(carteira_inicial)
saldo_final = sum(carteira)

lucro =  saldo_final - saldo_inicial
percentual_carteira = (saldo_final / saldo_inicial - 1)*100

print(saldo_inicial, '->', saldo_final, '\n', 'lucro: ', lucro, '\n', 'Empresas: ', investir_empresas)
print('percentual: ', percentual_carteira, '%')



# Comparando crescimento do BOVESPA com a CARTEIRA
variacao_BVSP = (df_BVSP.loc['2021-03-31', 'IBOV'] / df_BVSP.loc['2020-12-31', 'IBOV'] - 1) * 100
print('percentual carteira: ', percentual_carteira, '%')
print('percentual Bovespa: ', variacao_BVSP, '%')


# Armazenando IA em um arquivo
import joblib

joblib.dump(modelo_melhorado, 'ia_randomforest_carteira_acoes.jiblib')


modelo_ia = joblib.load('ia_randomforest_carteira_acoes.jiblib')
# Para prever proximas carteiras
# devemos criar um DF com 'Fornecedores', 'Outros Ativos Circulantes', 'Lucros/Prejuízos Acumulados', 'Resultado Antes Tributação/Participações', 'Ativo Total', 'Obrigações Sociais e Trabalhistas', 'Obrigações Fiscais', 'Custo de Bens e/ou Serviços Vendidos', 'Resultado da Equivalência Patrimonial', 'Tributos Diferidos' nos campos para cada uma das empresas naquele trimestre
# ao rodar: ' previsao = modelo_ia.predict(df) ' ele vai retornar uma lista onde 0 representa 'Vender' e dois 'Comprar'
# basta comparar essa lista com a lista de empresas no df