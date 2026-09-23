<h1 align="center">
  🎬 OmniWatch - Backend
</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.141.1-00a393.svg">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-Async-336791.svg">
  <img alt="LightGBM" src="https://img.shields.io/badge/Machine_Learning-LightGBM-orange.svg">
</p>

<p align="center">
  O <strong>Backend do OmniWatch</strong> é uma API robusta e assíncrona responsável por gerenciar a plataforma de rastreamento e recomendação de filmes e séries.
</p>

## 🚀 Visão Geral

Desenvolvido para oferecer uma experiência rápida e escalável, o backend serve como núcleo de dados para o ecossistema OmniWatch. Ele lida com gerenciamento de usuários, rastreamento de progresso (filmes e episódios assistidos), integração com APIs externas (TMDB/TVDB) e conta com um **motor de recomendações baseado em Machine Learning** integrado, treinando modelos em background para sugestões personalizadas.

## 🛠️ Tecnologias Utilizadas

A stack foi cuidadosamente escolhida para maximizar performance, produtividade e tipagem forte:

- **Framework Web:** [FastAPI](https://fastapi.tiangolo.com/) (Alta performance e geração automática de documentação Swagger/OpenAPI)
- **Linguagem:** Python 3.10+
- **Banco de Dados:** PostgreSQL acessado assincronamente via `asyncpg`
- **ORM & Migrations:** SQLAlchemy 2.0 e Alembic
- **Machine Learning:** LightGBM, Scikit-learn, e NumPy para geração de embeddings, clusters e rankeamento de recomendações.
- **Autenticação & Segurança:** JWT (JSON Web Tokens), `bcrypt` para hash de senhas, e `slowapi` para Rate Limiting.
- **Integração Externa:** `httpx` para consumo de APIs (TMDB e TVDB)
- **Gerenciador de Dependências:** Poetry

## ✨ Funcionalidades Principais

- **Autenticação e Autorização Segura:** Login/Cadastro com JWT e hash de senhas.
- **Motor de Recomendações (ML):** Sistema avançado que gera embeddings, classifica os gostos do usuário e sugere filmes e séries utilizando LightGBM.
- **Integração de APIs de Mídia:** Busca de filmes, séries, atores, diretores (Cast/Crew), posters e trailers em tempo real.
- **Sistema de Rastreamento (Tracking):**
  - Marcação de mídia como Assistido, Quero Assistir, ou Favorito.
  - Avaliação de mídias (Ratings em escala de estrelas).
- **Tarefas em Background Assíncronas:**
  - Sincronização automática do calendário de lançamentos.
  - Treinamento contínuo dos algoritmos de ML.
  - Cache de dados das coleções de mídias para leitura rápida.
- **Gerenciamento de Notificações:** Avisos sobre lançamentos esperados e atualizações.
- **Proteção contra Abusos:** Rate Limiting configurado em endpoints críticos.

## 📁 Estrutura do Projeto

O projeto segue os princípios de Domain-Driven Design (DDD) adaptados para FastAPI, onde cada domínio (ou módulo) da aplicação fica contido no seu próprio diretório.

```text
app/
├── auth/                 # Autenticação e tokens JWT
├── calendar/             # Sincronização de calendários de lançamentos
├── core/                 # Configurações gerais, banco de dados, rate limits
├── details/              # Detalhamento de filmes e séries
├── images/               # Cache ou manipulação de imagens (posters/backdrops)
├── media/ & tracking/    # Lógica central de mídias e progresso do usuário
├── person/               # Atores, diretores e equipes (Cast & Crew)
├── recommendation/       # Motor de IA: Rankers, embeddings e clustering
├── search/ & trending/   # Buscas avançadas e listas de "Em Alta"
└── users/                # Gerenciamento de perfil
```

## ⚙️ Como Executar Localmente

### 1. Pré-requisitos
- **Python 3.10+** instalado
- **Poetry** (Gerenciador de pacotes)
- Banco de dados **PostgreSQL** rodando localmente ou via Docker

### 2. Instalação

Clone este repositório e instale as dependências com o Poetry:

```bash
git clone https://github.com/itsJoaoVictor/Backend-OmniWatch.git
cd Backend-OmniWatch
poetry install
```

### 3. Configuração de Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto, usando o `.env.example` como base:

```env
DATABASE_URL="postgresql+asyncpg://postgres:sua_senha@localhost:5432/omniwatch"

# APIs de Terceiros (Necessário criar conta e gerar as chaves)
TMDB_API_KEY=sua_chave_tmdb_aqui
TVDB_API_KEY=sua_chave_tvdb_aqui

# Segurança
SECRET_KEY=sua_secret_key_aqui
```

### 4. Executando as Migrations do Banco de Dados

Crie as tabelas necessárias utilizando o Alembic:

```bash
poetry run alembic upgrade head
```

### 5. Iniciar o Servidor Web

```bash
poetry run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
A API estará disponível em `http://localhost:8000`. Você pode acessar a documentação interativa (Swagger UI) em `http://localhost:8000/docs`.

## 🤝 Autor

**João Victor**  
[GitHub](https://github.com/itsJoaoVictor)

---
*Projeto desenvolvido para fins de portfólio e aprimoramento de habilidades no ecossistema Python moderno e Machine Learning aplicados a backends reais.*