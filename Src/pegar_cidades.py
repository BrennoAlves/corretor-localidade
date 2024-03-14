import requests
import json
import os
import unicodedata

url_base = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios'

response = requests.get(url_base)
data = response.json()

nomes_municipios = [item['nome'] for item in data]

def normalizador(nome):
    return unicodedata.normalize('NFKD', nome.lower()).encode('ASCII', 'ignore').decode('utf-8')

dicionario_cidades = {normalizador(cidade): cidade for cidade in nomes_municipios}

output = os.path.join(os.path.dirname(__file__), '..', 'Data', 'cidades_canonicas.json')
with open(output, 'w', encoding='utf-8') as f:
    json.dump(dicionario_cidades, f, ensure_ascii = False, indent=4)
