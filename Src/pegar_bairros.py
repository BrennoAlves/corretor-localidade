import requests
from bs4 import BeautifulSoup
import json
import os
import time
import pandas as pd
import unidecode

CIDADES_CANONICAS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "cidades_canonicas.json")
BAIRROS_CANONICOS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "bairros_canonicos.json")

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"}

def obter_bairros_por_cidade(nome_cidade):
    
  nome_cidade_url = unidecode.unidecode(nome_cidade).lower().replace(' ', '-')  
  base_url = f"https://www.ruacep.com.br/mg/{nome_cidade_url}/bairros/"
  nomes_bairros = []

  with requests.Session() as session:
    pagina = 1
    while True:
      url_paginacao = f"{base_url}{pagina}/"
      response = session.get(url_paginacao, headers=headers)
      soup = BeautifulSoup(response.content, "html.parser")

      bairros = soup.find_all("div", class_="card-header")
      if not bairros:
        break

      for bairro in bairros:
        nome_bairro = bairro.find("strong").text.strip()
        nomes_bairros.append(nome_bairro)
      
      print(f"Página {pagina} de {nome_cidade} concluída.")
      pagina += 1
      time.sleep(1)

  return nomes_bairros

def obter_cidades_do_dataframe(df_imoveis):
    return df_imoveis["cidade_estimada"].unique().tolist()


def main(df_imoveis): # Adicione df_imoveis como argumento
    
    print()

    cidades_no_df = obter_cidades_do_dataframe(df_imoveis)

    bairros_canonicos = {}
    for cidade in cidades_no_df:
        if pd.isnull(cidade):
            continue
        bairros_canonicos[cidade] = obter_bairros_por_cidade(cidade)

    with open(BAIRROS_CANONICOS_JSON, "w", encoding="utf-8") as f:
        json.dump(bairros_canonicos, f, ensure_ascii=False, indent=2)

    print("Bairros canônicos salvos com sucesso!")

if __name__ == "__main__":
    main()