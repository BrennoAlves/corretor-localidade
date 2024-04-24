import json
import os
import pickle
import unidecode
from tqdm import tqdm
import numpy as np  
import pandas as pd  
from sentence_transformers import SentenceTransformer
import subprocess
import faiss  

import pegar_bairros


#Patchs
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = "embeddings"
PATH_VENV = "venv/Scripts/python.exe"
SALVAR_PATH = "processados/imoveis_processados.xlsx"
SCRIPT_PEGAR_CIDADES = "src/pegar_cidades.py"

#Variáveis globais
CIDADES_CANONICAS_JSON = os.path.join(DATA_PATH, "cidades_canonicas.json")
IMOVEIS_JSON = os.path.join(DATA_PATH, "imoveis.json")
BAIRROS_CANONICOS_JSON = os.path.join(DATA_PATH, "bairros_canonicos.json")
CIDADES_CANONICAS_PKL = os.path.join(EMBEDDINGS_PATH, "cidades_canonicas.pkl")
BAIRROS_CANONICOS_PKL = os.path.join(EMBEDDINGS_PATH, "bairros_canonicos.pkl")
MODALIDADES = {1: "Venda", 2: "Aluguel", 3: "Temporada"}
STATUS = {1: "Ativo", 2: "Inativo"}



#Limpa o terminal, tenho TOC
def limpar_console():
    os.system("clear" if os.name == "posix" else "cls")


#Carrega os arquivos necessários para rodar o script
def carregar_dados(imoveis_json, modalidades, status):
    
    verificar_cidades_canonicas(CIDADES_CANONICAS_JSON)
    cidades_canonicas: dict[str, list[str]] = json.load(open(CIDADES_CANONICAS_JSON, "r", encoding="utf-8"))
    
    #Carregar df_imoveis ANTES de verificar bairros canônicos
    modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df_imoveis = pd.read_json(imoveis_json)
    df_imoveis = df_imoveis[~df_imoveis["id"].isin([0, 1, 14, 15])]
    df_imoveis = df_imoveis.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df_imoveis["id_modalidade"] = df_imoveis["id_modalidade"].replace(modalidades)
    df_imoveis["status"] = df_imoveis["status"].replace(status)
    df_imoveis = df_imoveis.replace("", np.nan).fillna(np.nan).dropna(subset=["cidade"])
    df_imoveis["cidade"] = df_imoveis["cidade"].astype("str")

    return df_imoveis, cidades_canonicas, modelo


#Verifica se o arquivo cidades_canonicas.json existe, se não roda o script pegar_cidades.py para montar dicionario de cidades do BR
def verificar_cidades_canonicas(arquivo: str) -> None:
    if not os.path.exists(arquivo):
        print(f"A lista de cidades canônicas não foi encontrada. Consultando a API do IBGE.")
        try:
            _ = subprocess.run([PATH_VENV, 'Src/pegar_cidades.py'])
        except Exception as e:
            print(f"Ocorreu um erro ao executar o script: {e}")
    else:
        print("A lista de cidades canônicas foi encontrada.")


#Normaliza o nome das cidades, usando LLM para comparar semanticamente os dados brutos e o dicionário de cidades_canonicas, usando Faiss para gerar índice e agilizar essa comparação e não ter que percorrer toda a lista
def normalizar_cidades(df_imoveis, cidades_canonicas, modelo, embeddings_path):

    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)

    try:
        embeddings_cidades_canonicas = pickle.load(open(os.path.join(embeddings_path, "cidades_canonicas.pkl"), "rb"))
    except:
        embeddings_cidades_canonicas = {}
        for cidade_normalizada, _ in cidades_canonicas.items():
            embeddings_cidades_canonicas[cidade_normalizada] = modelo.encode(cidade_normalizada, convert_to_tensor=True, device="cuda")
        pickle.dump(embeddings_cidades_canonicas, open(os.path.join(embeddings_path, "cidades_canonicas.pkl"), "wb"))

    #Criar índice Faiss para cidades canônicas
    index_cidades = faiss.IndexFlatL2(list(embeddings_cidades_canonicas.values())[0].shape[0])
    index_cidades.add(np.stack([t.cpu().numpy() for t in embeddings_cidades_canonicas.values()]))

    #Vetorização em lote para todas as cidades
    embeddings_cidades = modelo.encode(df_imoveis["cidade"].tolist(), batch_size=64, convert_to_tensor=True, device="cuda")

    for contador, (index, linha) in tqdm(enumerate(df_imoveis.iterrows()), total=len(df_imoveis)):
        cidade = linha["cidade"]
        if pd.isnull(cidade):
            continue
        embedding_cidade = embeddings_cidades[contador]
        _, indices = index_cidades.search(np.expand_dims(embedding_cidade.cpu(), axis=0), k=1)
        cidade_corrigida = list(embeddings_cidades_canonicas.keys())[indices[0][0]]
        cidade_corrigida = unidecode.unidecode(cidade_corrigida).title()
        df_imoveis.at[index, "cidade_estimada"] = cidade_corrigida
    
    return df_imoveis


def verificar_bairros_canonicos(arquivo: str, df_imoveis) -> None:

    if not os.path.exists(arquivo):
        print(f"A lista de bairros canônicos não foi encontrada. Obtendo bairros...")
        try:
            pegar_bairros.main(df_imoveis)
            bairros_canonicos: dict[str, list[str]] = json.load(open(BAIRROS_CANONICOS_JSON, "r", encoding="utf-8"))

        except Exception as e:
            print(f"Ocorreu um erro ao obter os bairros: {e}")
    else:
        print("A lista de bairros canônicos foi encontrada.")
        bairros_canonicos: dict[str, list[str]] = json.load(open(BAIRROS_CANONICOS_JSON, "r", encoding="utf-8"))

    return bairros_canonicos

#Normaliza o nome dos bairros, usando LLM para comparar semanticamente os dados brutos e o dicionário de bairros_canonicos, usando Faiss para gerar índice e agilizar essa comparação e não ter que percorrer toda a lista
def normalizar_bairros(df_imoveis, bairros_canonicos, modelo, embeddings_path): 

    try:
        embeddings_bairros_canonicos = pickle.load(open(os.path.join(embeddings_path, "bairros_canonicos.pkl"), "rb"))
    except:
        embeddings_bairros_canonicos = {}
        for bairro_normalizado, _ in bairros_canonicos.items():
            embeddings_bairros_canonicos[bairro_normalizado] = modelo.encode(bairro_normalizado, convert_to_tensor=True, device="cuda")
        pickle.dump(embeddings_bairros_canonicos, open(os.path.join(embeddings_path, "bairros_canonicos.pkl"), "wb"))

    index_bairros = faiss.IndexFlatL2(list(embeddings_bairros_canonicos.values())[0].shape[0])
    index_bairros.add(np.stack([t.cpu().numpy() for t in embeddings_bairros_canonicos.values()]))

    embeddings_bairros = modelo.encode(
        df_imoveis["bairro"].astype(str).tolist(), batch_size=64, convert_to_tensor=True, device="cuda")

    for contador, (index, linha) in tqdm(enumerate(df_imoveis.iterrows()), total=len(df_imoveis)):
        bairro = linha["bairro"]
        if pd.isnull(bairro):
            continue
        embedding_bairro = embeddings_bairros[contador]
        _, indices = index_bairros.search(np.expand_dims(embedding_bairro.cpu(), axis=0), k=1)
        bairro_estimado = list(embeddings_bairros_canonicos.keys())[indices[0][0]]
        df_imoveis.at[index, "bairro_estimado"] = bairro_estimado

    return df_imoveis


#Corrigindo a escrita do nome das cidades batendo de novo no cidades_canonicas
def corrigir_nomes_cidades(df_imoveis, cidades_canonicas):
    #Transformar a coluna 'cidade_estimada' para minúsculas e remover espaços duplicados no final do nome
    df_imoveis['cidade_estimada'] = df_imoveis['cidade_estimada'].str.lower().str.rstrip()

    #Comparar cada cidade estimada com as cidades canônicas
    for index, row in df_imoveis.iterrows():
        cidade_estimada = row['cidade_estimada']

        #Verificar se a cidade estimada está no dicionário de cidades canônicas
        if cidade_estimada in cidades_canonicas:

            #Se estiver, substituir pelo nome canônico correspondente
            cidade_canonica = cidades_canonicas[cidade_estimada]
            df_imoveis.at[index, 'cidade_estimada'] = cidade_canonica

    return df_imoveis


#Salva o resultando em arquivo excel
def salvar_resultados(df_imoveis, output_file):
    if not os.path.exists("processados"):
        os.makedirs("processados")
    df_imoveis.to_excel(output_file)
    

if __name__ == "__main__":
    #Pipeline de execução
    print("Iniciando processo.")
    print("Carregando dados.")

    df_imoveis, cidades_canonicas, modelo = carregar_dados(IMOVEIS_JSON, MODALIDADES, STATUS)

    print("Iniciando processo de correção dos nomes das cidades.")

    df_imoveis = normalizar_cidades(df_imoveis, cidades_canonicas, modelo, EMBEDDINGS_PATH)

    print("Iniciando processo de correção dos nomes dos bairros.")

    bairros_canonicos = verificar_bairros_canonicos(BAIRROS_CANONICOS_JSON, df_imoveis)
    df_imoveis = normalizar_bairros(df_imoveis, bairros_canonicos, modelo, EMBEDDINGS_PATH)

    print("Fazendo últimas correções.")
    df_imoveis = corrigir_nomes_cidades(df_imoveis, cidades_canonicas)
    salvar_resultados(df_imoveis, SALVAR_PATH)

    print("Finalizado! Salvo em /processados/")