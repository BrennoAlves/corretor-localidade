import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer, util
import pickle

os.system('clear' if os.name == 'posix' else 'cls')
print('Pré-processamento')

tipos = pd.read_csv('tipos.csv').drop(columns=['sigla','id_tipo_terra'])

descarte = ['id_proprietario', 'posicao', 'sacadas', 'video', 'id_filial',
                'id_captador', 'id_captador2', 'referencia', 'obs', 'descricao', 'destaque',
                'chave', 'exclusivo', 'parceria', 'placa', 'ids_atributos', 'comissao_porcentagem',
                'comissao_valor', 'id_categoria_extra', 'testada', 'pais', 'id_usuario', 'data_alteracao',
                'lado_direito', 'lado_esquerdo', 'fundos', 'area_servicos', 'copas', 'clicks',
                'codigo_imobiliaria', 'vivareal', 'zapimoveis', 'terra', 'olx', 'mostrar_condominio',
                'ordem', 'restricoes', 'ultima_alteracao', 'id_municipio', 'id_bairro', 'n_iptu',
                'n_matricula', 'n_damae', 'n_cliente_cemig', 'n_instalacao_cemig', 'area_gourmet',
                'lavabo', 'lavanderia', 'closet', 'piscina', 'a_partir','id_tipo',
                'mostrar_iptu', 'id', 'id_cliente']

cidades_canonicas = json.load(open("cidades_canonicas.json", "r", encoding="utf-8"))
modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = pd.read_json('imoveis.json')
df = df[~df['id'].isin([0, 1, 14, 15, 19, 23, 27, 28, 35, 36, 44, 46, 47, 48, 52])]
df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

for i in range(len(df)):
    try:
        df.loc[i, "nome_tipo"] = tipos.loc[tipos["id"] == df.loc[i, "id_tipo"], "nome"].iloc[0]
    except:
        pass

df = df.dropna(subset=['nome_tipo']).drop(columns=descarte)

df.loc[df['id_modalidade'] == 1, 'id_modalidade'] = 'Venda'
df.loc[df['id_modalidade'] == 2, 'id_modalidade'] = 'Aluguel'
df.loc[df['id_modalidade'] == 3, 'id_modalidade'] = 'Temporada'

df.loc[df['status'] == 1, 'status'] = 'Ativo'
df.loc[df['status'] == 2, 'status'] = 'Inativo'

df = df.replace('', np.nan).fillna(np.nan).dropna(subset=["cidade"])

df["cidade"] = df["cidade"].astype("str")

os.system('clear' if os.name == 'posix' else 'cls')

#EXCLUIR DEPOIS DE TESTAR
df = df.head(1000)

# Corração das cidades

embeddings_cidades_canonicas = {}

for cidade_normalizada, cidade_original in cidades_canonicas.items():
    embeddings_cidades_canonicas[cidade_normalizada] = modelo.encode(cidade_normalizada, convert_to_tensor=True, device='cuda')
pickle.dump(embeddings_cidades_canonicas, open('embeddings/cidades_canonicas.pkl', 'wb'))

for contador, (index, linha) in enumerate(df.iterrows()):
    progresso = (contador / len(df)) * 100
    print(f'Loop {contador+1}/{len(df)} - {progresso:.2f}%')
    cidade = linha["cidade"]

    if pd.isnull(cidade):
        continue

    max_sim = 0
    cidade_corrigida = cidade
    embedding_cidade = modelo.encode(cidade, convert_to_tensor=True, device='cuda')

    for cidade_normalizada, embedding_cidade_normalizada in embeddings_cidades_canonicas.items():
        sim = util.pytorch_cos_sim(embedding_cidade, embedding_cidade_normalizada)

        if sim > max_sim:
            max_sim = sim
            cidade_corrigida = cidades_canonicas[cidade_normalizada]

    df.at[index, "cidade_estimada"] = cidade_corrigida

# Correção de bairros (apenas para São João del Rei)

df = df.query('cidade_estimada == "São João del Rei"')

if os.path.exists('embeddings/bairros_canonicos.pkl'):
    embeddings_bairros_canonicos = pickle.load(open('embeddings/bairros_canonicos.pkl', 'rb'))
else:
    bairros_canonicos = json.load(open('bairros_canonicos.json', 'r', encoding='utf-8'))
    embeddings_bairros_canonicos = {}
    for bairro_info in bairros_canonicos:
        bairro = bairro_info["bairro"]
        embeddings_bairros_canonicos[bairro] = modelo.encode(bairro, convert_to_tensor=True, device='cuda')
    pickle.dump(embeddings_bairros_canonicos, open('embeddings/bairros_canonicos.pkl', 'wb'))

def similaridade_bairros(embedding_1, embedding_2):
    return util.pytorch_cos_sim(embedding_1, embedding_2)

for contador, (index, linha) in enumerate(df.iterrows()):
    progresso = (contador / len(df)) * 100
    print(f'Loop {contador+1}/{len(df)} - {progresso:.2f}%')
    bairro = linha["bairro"]

    if pd.isnull(bairro):
        continue

    max_sim = 0
    bairro_corrigido = bairro
    embedding_bairro = modelo.encode(bairro, convert_to_tensor=True, device='cuda')

    for bairro_normalizado, embedding_bairro_normalizado in embeddings_bairros_canonicos.items():
        sim = similaridade_bairros(embedding_bairro, embedding_bairro_normalizado)

        if sim > max_sim:
            max_sim = sim
            bairro_corrigido = bairro_normalizado

    df.at[index, "bairro_corrigido"] = bairro_corrigido

# Salvando o DataFrame

df.to_excel('bairros_finalizado.xlsx')

print('Finalizado')