import pandas as pd
import numpy as np


df = pd.read_json('ids.json')

ex_clientes = [0, 1, 14, 15, 19, 23, 27, 28, 35, 36, 44, 46, 47, 48, 52]
df = df[~df['id'].isin(ex_clientes)]

df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')

tipos = pd.read_csv('tipos.csv')
tipos = tipos.drop(columns=['sigla','id_tipo_terra'])

df = df.dropna(subset=['id_tipo'])

for i in range(len(df)):
    try:
        id_tipo = df.loc[i, "id_tipo"]
        nome_tipo = tipos.loc[tipos["id"] == id_tipo, "nome"].iloc[0]
        df.loc[i, "nome_tipo"] = nome_tipo
    except:
        pass

df = df.dropna(subset=['nome_tipo'])

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

df = df.drop(columns=descarte)

df.loc[df['id_modalidade'] == 1, 'id_modalidade'] = 'Venda'
df.loc[df['id_modalidade'] == 2, 'id_modalidade'] = 'Aluguel'
df.loc[df['id_modalidade'] == 3, 'id_modalidade'] = 'Temporada'

df.loc[df['status'] == 1, 'status'] = 'Ativo'
df.loc[df['status'] == 2, 'status'] = 'Inativo'

df = df.replace('', np.nan).fillna(np.nan)
df = df.replace('', np.nan)
df = df.fillna(np.nan)

df["cidade"] = df["cidade"].astype("str")

print('ok')