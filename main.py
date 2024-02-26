import pandas as pd
import numpy as np
import json
import os
import pickle
from sentence_transformers import SentenceTransformer, util

def limpar_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def carregar_e_preprar_dados():
    limpar_console()
    print('Carregando e preparando dados...')

    tipos_df = pd.read_csv('tipos.csv').drop(columns=['sigla','id_tipo_terra'])
    colunas_descartadas = ['id_proprietario', 'posicao', 'sacadas', 'video', 'id_filial',
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

    df_imoveis = pd.read_json('imoveis.json')
    df_imoveis = df_imoveis[~df_imoveis['id'].isin([0, 1, 14, 15, ...])]
    df_imoveis = df_imoveis.dropna(axis=0, how='all').dropna(axis=1, how='all')

    for i in range(len(df_imoveis)):
        try:
            df_imoveis.loc[i, "tipo_imovel"] = tipos_df.loc[tipos_df["id"] == df_imoveis.loc[i, "id_tipo"], "nome"].iloc[0]
        except:
            pass

    df_imoveis = df_imoveis.dropna(subset=['tipo_imovel']).drop(columns=colunas_descartadas)

    modalidades = {1: 'Venda', 2: 'Aluguel', 3: 'Temporada'}
    df_imoveis['id_modalidade'] = df_imoveis['id_modalidade'].replace(modalidades)

    status = {1: 'Ativo', 2: 'Inativo'}
    df_imoveis['status'] = df_imoveis['status'].replace(status)

    df_imoveis = df_imoveis.replace('', np.nan).fillna(np.nan).dropna(subset=["cidade"])
    df_imoveis["cidade"] = df_imoveis["cidade"].astype("str")

    # REMOVER APÓS TESTE
    df_imoveis = df_imoveis.head(1000)

    return df_imoveis, cidades_canonicas, modelo

def normalizar_cidades(df_imoveis, cidades_canonicas, modelo):
    print('Iniciando processo de correção dos nomes das cidades')

    embeddings_cidades_canonicas = {}

    for cidade_normalizada, cidade_original in cidades_canonicas.items():
        embeddings_cidades_canonicas[cidade_normalizada] = modelo.encode(cidade_normalizada, convert_to_tensor=True, device='cuda')

    pickle.dump(embeddings_cidades_canonicas, open('embeddings/cidades_canonicas.pkl', 'wb'))

def similaridade_cidades(embedding_1, embedding_2):
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

def normalizar_cidades(df_imoveis, cidades_canonicas, modelo):
    print('Iniciando processo de correção dos nomes das cidades')

    embeddings_cidades_canonicas = {}

    for cidade_normalizada, cidade_original in cidades_canonicas.items():
        embeddings_cidades_canonicas[cidade_normalizada] = modelo.encode(cidade_normalizada, convert_to_tensor=True, device='cuda')

    pickle.dump(embeddings_cidades_canonicas, open('embeddings/cidades_canonicas.pkl', 'wb'))

    def similaridade_cidades(embedding_1, embedding_2):
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

    for contador, (index, linha) in enumerate(df_imoveis.iterrows()):
        progresso = (contador / len(df_imoveis)) * 100
        print(f'Loop {contador+1}/{len(df_imoveis)} - {progresso:.2f}%')
        cidade = linha["cidade"]

        if pd.isnull(cidade):
            continue

        max_sim = 0
        cidade_corrigida = cidade
        embedding_cidade = modelo.encode(cidade, convert_to_tensor=True, device='cuda')

        for cidade_normalizada, embedding_cidade_normalizada in embeddings_cidades_canonicas.items():
            sim = similaridade_cidades(embedding_cidade, embedding_cidade_normalizada)

            if sim > max_sim:
                max_sim = sim
                cidade_corrigida = cidades_canonicas[cidade_normalizada]

        df_imoveis.at[index, "cidade_estimada"] = cidade_corrigida

    return df_imoveis

def normalizar_bairros(df_imoveis, modelo):
    print('Iniciando processo de correção dos nomes dos bairros (para São João del Rei)')
    df_imoveis = df_imoveis.query('cidade_estimada == "São João del Rei"')

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
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

    for contador, (index, linha) in enumerate(df_imoveis.iterrows()):
        progresso = (contador / len(df_imoveis)) * 100
        print(f'Loop {contador+1}/{len(df_imoveis)} - {progresso:.2f}%')
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

        df_imoveis.at[index, "bairro_corrigido"] = bairro_corrigido

    return df_imoveis

def salvar_dados_finalizados(df_imoveis):
    df_imoveis.to_excel('bairros_finalizado.xlsx')
    print('Finalizado')

if __name__ == "__main__":
    df_imoveis, cidades_canonicas, modelo = carregar_e_preprar_dados()
    df_imoveis = normalizar_cidades(df_imoveis, cidades_canonicas, modelo)
    df_imoveis = normalizar_bairros(df_imoveis, modelo)
    salvar_dados_finalizados(df_imoveis)