import faiss  
import json
import os
import pickle
import unidecode
from tqdm import tqdm
import numpy as np  
import pandas as pd  
from sentence_transformers import SentenceTransformer
import subprocess
import pegar_cidades

DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "Data")
EMBEDDINGS_PATH: str = "Embeddings"
TIPOS_CSV: str = os.path.join(DATA_PATH, "tipos.csv")
CIDADES_CANONICAS_JSON: str = os.path.join(DATA_PATH, "cidades_canonicas.json")
IMOVEIS_JSON: str = os.path.join(DATA_PATH, "imoveis.json")
BAIRROS_CANONICOS_JSON: str = os.path.join(DATA_PATH, "bairros_canonicos.json")
CIDADES_CANONICAS_PKL: str = os.path.join(EMBEDDINGS_PATH, "cidades_canonicas.pkl")
BAIRROS_CANONICOS_PKL: str = os.path.join(EMBEDDINGS_PATH, "bairros_canonicos.pkl")

MODALIDADES: dict[int, str] = {1: 'Venda', 2: 'Aluguel', 3: 'Temporada'}
STATUS: dict[int, str] = {1: 'Ativo', 2: 'Inativo'}
SCRIPT_PEGAR_CIDADES: str = 'pegar_cidades.py'
VENV_CAMINHO: str = "venv/Scripts/python.exe"


def limpar_console():
    os.system('clear' if os.name == 'posix' else 'cls')


def carregar() -> tuple[pd.DataFrame, dict[str, list[str]], SentenceTransformer]:
    limpar_console()
    print('Carregando e preparando dados...')
    tipos_df: pd.DataFrame = pd.read_csv(TIPOS_CSV).drop(columns=['sigla', 'id_tipo_terra'])
    verificar_cidades_canonicas(CIDADES_CANONICAS_JSON)
    cidades_canonicas: dict[str, list[str]] = json.load(open(CIDADES_CANONICAS_JSON, "r", encoding="utf-8"))
    modelo: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    df_imoveis: pd.DataFrame = pd.read_json(IMOVEIS_JSON)
    df_imoveis = df_imoveis[~df_imoveis['id'].isin([0, 1, 14, 15])]
    df_imoveis = df_imoveis.dropna(axis=0, how='all').dropna(axis=1, how='all')
    for i in range(len(df_imoveis)):
        try:
            df_imoveis.loc[i, "tipo_imovel"] = tipos_df.loc[tipos_df["id"] == df_imoveis.loc[i, "id_tipo"], "nome"].iloc[0]
        except:
            pass
    df_imoveis['id_modalidade'] = df_imoveis['id_modalidade'].replace(MODALIDADES)
    df_imoveis['status'] = df_imoveis['status'].replace(STATUS)
    df_imoveis = df_imoveis.replace('', np.nan).fillna(np.nan).dropna(subset=["cidade"])
    df_imoveis["cidade"] = df_imoveis["cidade"].astype("str")
    return df_imoveis, cidades_canonicas, modelo


def verificar_cidades_canonicas(arquivo: str) -> None:
    if not os.path.exists(arquivo):
        print(f"A lista de cidades canônicas não foi encontrada. Consultando a API do IBGE")
        try:
            _ = subprocess.run([VENV_CAMINHO, 'Src/pegar_cidades.py'])
        except Exception as e:
            print(f"Ocorreu um erro ao executar o script: {e}")
    else:
        print("A lista de cidades canônicas foi encontrada.")


def normalizar_cidades(df_imoveis: pd.DataFrame, cidades_canonicas: dict[str, list[str]], modelo: SentenceTransformer) -> pd.DataFrame:
    print('Iniciando processo de correção dos nomes das cidades')
    try:
        embeddings_cidades_canonicas: dict[str, faiss.swigfaiss.Float32Vector] = pickle.load(open(CIDADES_CANONICAS_PKL, 'rb'))
    except:
        embeddings_cidades_canonicas = {}
        for cidade_normalizada, _ in cidades_canonicas.items():
            embeddings_cidades_canonicas[cidade_normalizada] = modelo.encode(
                cidade_normalizada, convert_to_tensor=True, device='cuda')
        pickle.dump(embeddings_cidades_canonicas, open(CIDADES_CANONICAS_PKL, 'wb'))

    #Criar índice Faiss para cidades canônicas
    index_cidades = faiss.IndexFlatL2(list(embeddings_cidades_canonicas.values())[0].shape[0])
    index_cidades.add(np.stack([t.cpu().numpy() for t in embeddings_cidades_canonicas.values()]))

    #Vetorização em lote para todas as cidades
    embeddings_cidades = modelo.encode(df_imoveis["cidade"].tolist(), batch_size=64, convert_to_tensor=True, device='cuda')

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


def normalizar_bairros(df_imoveis: pd.DataFrame, modelo: SentenceTransformer) -> pd.DataFrame:
    print('Iniciando processo de correção dos nomes dos bairros (para São João del Rei)')
       
    nomes_municipios: list[str] = pegar_cidades.obter_nomes_municipios()
    dicionario_cidades: dict[str, str] = pegar_cidades.criar_dicionario_cidades(nomes_municipios)
    pegar_cidades.salvar_dicionario_cidades(dicionario_cidades, CIDADES_CANONICAS_JSON)

    df_bairros_sjr = df_imoveis.query('cidade_estimada == "Sao Joao Del Rei"')

    if os.path.exists(BAIRROS_CANONICOS_PKL):
        embeddings_bairros_canonicos = pickle.load(open(BAIRROS_CANONICOS_PKL, 'rb'))
    else:
        bairros_canonicos = json.load(open(BAIRROS_CANONICOS_JSON, 'r', encoding='utf-8'))
        embeddings_bairros_canonicos = {}
        for bairro_info in bairros_canonicos:
            bairro = bairro_info["bairro"]
            embeddings_bairros_canonicos[bairro] = modelo.encode(bairro, convert_to_tensor=True, device='cuda')
        pickle.dump(embeddings_bairros_canonicos, open(BAIRROS_CANONICOS_PKL, 'wb'))

    #Criar índice Faiss para cidades canônicas
    index_bairros = faiss.IndexFlatL2(list(embeddings_bairros_canonicos.values())[0].shape[0])
    index_bairros.add(np.stack([t.cpu().numpy() for t in embeddings_bairros_canonicos.values()]))

    #Vetorização em lote para todas as cidades
    embeddings_bairros = modelo.encode(df_bairros_sjr["bairro"].astype(str).tolist(), batch_size=64, convert_to_tensor=True, device='cuda')

    for contador, (index, linha) in tqdm(enumerate(df_bairros_sjr.iterrows()), total=len(df_bairros_sjr)):
        bairro = linha["bairro"]
        if pd.isnull(bairro):
            continue
        embedding_bairro = embeddings_bairros[contador]
        _, indices = index_bairros.search(np.expand_dims(embedding_bairro.cpu(), axis=0), k=1)
        bairro_corrigido = list(embeddings_bairros_canonicos.keys())[indices[0][0]]
        df_imoveis.at[index, "bairro_corrigido"] = bairro_corrigido
    return df_imoveis


def salvar(df_imoveis: pd.DataFrame) -> None:
    df_imoveis.to_excel('processados/bairros_finalizado.xlsx')
    print('Finalizado')


if __name__ == "__main__":
    df_imoveis, cidades_canonicas, modelo = carregar()
    df_imoveis = normalizar_cidades(df_imoveis, cidades_canonicas, modelo)
    df_imoveis = normalizar_bairros(df_imoveis, modelo)
    salvar(df_imoveis)