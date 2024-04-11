## README

Este projeto se destina a realizar a normalização dos nomes das cidades e bairros em um conjunto de dados de imóveis exportado do Homeclix. Ele segue os seguintes passos:

### 1.1. Carregamento e Preparação dos Dados:
   - Carrega um conjunto de dados contendo informações sobre os tipos de imóveis e cidades canônicas.
   - Carrega o modelo de incorporação de frases SentenceTransformer.

### 1.2. Normalização das Cidades:
   - Gera representações de incorporação para as cidades canônicas usando o modelo SentenceTransformer.
   - Calcula a similaridade de cosseno entre cada cidade no conjunto de dados e as cidades canônicas.
   - Corrige os nomes das cidades no conjunto de dados com base na cidade canônica mais similar.

### 1.3. Normalização dos Bairros (para São João del Rei):
   - Filtra o conjunto de dados para conter apenas imóveis localizados em São João del Rei.
   - Gera representações de incorporação para os bairros canônicos em São João del Rei usando o modelo SentenceTransformer.
   - Calcula a similaridade de cosseno entre cada bairro no conjunto de dados e os bairros canônicos.
   - Corrige os nomes dos bairros no conjunto de dados com base no bairro canônico mais similar.

### 1.4. Salvamento dos Dados:
   - Salva o conjunto de dados resultante em um arquivo Excel chamado "bairros_finalizado.xlsx".

# Configuração do Ambiente

Este guia fornece instruções sobre como configurar um ambiente Python para trabalhar com o projeto em questão.

## Passo 1: Criar uma Virtual Environment (venv)

1. Abra um terminal ou prompt de comando.
2. Navegue até o diretório onde deseja criar sua virtual environment.
3. Execute o seguinte comando para criar uma nova virtual environment chamada "." (ponto) ou seja criativo:

    ```
    python -m venv .
    ```

4. Ative a virtual environment:

    - No Windows:

        ```
        .\Scripts\activate
        ```

    - No macOS e Linux:

        ```
        source ./bin/activate
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
