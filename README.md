## README

Este projeto realiza a normalização dos nomes das cidades e bairros em um conjunto de dados, ele deve funcionar em qualquer conjunto caso tenha as colunas bairro e cidade, contendo seus respesctivos nomes salvos como string. Ele segue os seguintes passos:

### 1.1. Carregamento e Preparação dos Dados:
   - Carrega um conjunto de dados contendo informações sobre os tipos de imóveis e cidades canônicas.
   - Carrega o modelo de incorporação de frases SentenceTransformer.

### 1.2. Normalização das Cidades:
   - Gera representações de incorporação para as cidades canônicas usando o modelo SentenceTransformer.
   - Calcula a similaridade de cosseno entre cada cidade no conjunto de dados e as cidades canônicas.
   - Corrige os nomes das cidades no conjunto de dados com base na cidade canônica mais similar.

### 1.3. Normalização dos Bairros:
   - Calcula a similaridade de cosseno entre cada bairro no conjunto de dados e os bairros canônicos.
   - Corrige os nomes dos bairros no conjunto de dados com base no bairro canônico mais similar.

### 1.4. Salvamento dos Dados:
   - Salva o conjunto de dados resultante em um arquivo Excel chamado "bairros_finalizado.xlsx".

# Configuração do Ambiente

## Passo 1: Criar uma Virtual Environment (venv)

1. Abra um terminal ou prompt de comando.
2. Navegue até o diretório onde deseja criar sua virtual environment.
3. Execute o seguinte comando para criar uma nova virtual environment chamada "venv" ou seja criativo:

    ```
    python -m venv venv
    ```

4. Ative a virtual environment:

    - No Windows:

        ```
        venv\Scripts\activate
        ```

    - No macOS e Linux:

        ```
        source venv/bin/activate
        ```

## Passo 2: Instalar os Requisitos do Projeto

1. Com a virtual environment ativada, instale os requisitos do projeto a partir do arquivo `requirements.txt`:

    ```
    pip install -r requirements.txt
    ```

## Passo 3: Instalar o PyTorch

1. Após instalar os requisitos do projeto, você pode instalar o PyTorch manualmente, caso necessário (vai ser):

    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    Certifique-se de estar dentro da virtual environment antes de executar este comando.

## Passo 4: Execução

Basicamente só rodar o arquivo main.py tendo os dados com nome de imoveis.json (que possua as colunas bairro e cidade, pode ter mais, não vai interferir, mas essas são obrigatórias) em uma pasta com nome de data, ele deve buscar os dados necessários e salvar em arquivos tanto na pasta data quanto embeddings.

Para dados que não estejam em json e/ou em colunas com nomes diferentes as mudanças são bem simples, basta mudar nome de variáveis e o método que o pandas usa para abrir o arquivo.


