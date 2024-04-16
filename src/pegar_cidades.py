import json
import os
import unicodedata
import requests

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data')
CIDADES_CANONICAS_JSON = os.path.join(DATA_PATH, 'cidades_canonicas.json')
URL_BASE = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios'

def obter_nomes_municipios():
    response = requests.get(URL_BASE)
    data = response.json()
    return [item['nome'] for item in data]

def normalizar_nome(nome):
    return unicodedata.normalize('NFKD', nome.lower()).encode('ASCII', 'ignore').decode('utf-8')

def criar_dicionario_cidades(nomes_municipios):
    return {normalizar_nome(cidade): cidade for cidade in nomes_municipios}

def salvar_dicionario_cidades(dicionario_cidades, arquivo_saida):
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(dicionario_cidades, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    nomes_municipios = obter_nomes_municipios()
    dicionario_cidades = criar_dicionario_cidades(nomes_municipios)
    salvar_dicionario_cidades(dicionario_cidades, CIDADES_CANONICAS_JSON)