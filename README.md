# HACKATHON – Classificação WEEE com Multiagentes

Breve descrição
- Projeto para classificar imagens de resíduos de EEE (WEEE) em 6 categorias usando Azure AI Vision e um orquestrador opcional de agentes (Azure AI Agents). A lógica de visão está em [`app.services.vision`](app/services/vision.py), o classificador WEEE em [`app.agents.weee_classifier`](app/agents/weee_classifier.py) e as rotas Flask em [app/routes.py](app/routes.py).

Contribuidores
- Substitua pelos nomes/e-mails reais antes do primeiro commit:
  - Vitoriano Ferrero Martin Junior <victorianojr@gmail.com>
  - Lilian Ferreira <beltrano@example.com>
  - Felipe Lima Fuhrmann <felipe.fuhrmann@outlook.com>

Configuração (rápido)
1. Copie e edite variáveis sensíveis:
   - cp .env.example .env
   - Preencha chaves Azure: AI_SERVICE_ENDPOINT, AI_SERVICE_KEY
   - (Opcional) AGENTS_PROJECT_ENDPOINT, AGENTS_MODEL_DEPLOYMENT para o árbitro LLM
   - Configure números/tokens para WhatsApp/Telegram conforme `.env.example`
   - Veja `.env.example` em [`.env.example`](.env.example)
2. Crie e ative venv, instale dependências:
   - python -m venv .venv
   - Windows: .\.venv\Scripts\Activate.ps1
   - macOS/Linux: source .venv/bin/activate
   - pip install -r [requirements.txt](requirements.txt)
3. (Opcional) Azure CLI: az login para DefaultAzureCredential se usar Azure AI Agents.

Executando localmente
- Iniciar servidor Flask:
  - Windows PowerShell:
    $env:FLASK_APP="app"; $env:FLASK_ENV="development"; flask run
  - Ou: python -m flask run
- Para expor localmente, use Cloudflare Tunnel / ngrok e defina PUBLIC_BASE_URL no .env (ex.: cloudflared tunnel --url http://localhost:5000).

Endpoints principais (ver implementação em [app/routes.py](app/routes.py))
- UI: GET /
- Análise visual: POST /api/analyze
- Classificação WEEE: POST /api/classify  (use ?llm=true para forçar árbitro LLM por requisição)
- Webhooks: /twilio/whatsapp, /meta/whatsapp, /telegram/webhook

Testes
- Executar testes unitários:
  - pytest
  - Caso de exemplo: [tests/test_vision.py](tests/test_vision.py)

Boas práticas de commits — Conventional Commits (resumo)
- Formato: <tipo>[escopo opcional]: <descrição curta>
- Tipos comuns:
  - feat: nova funcionalidade
  - fix: correção de bug
  - docs: documentação
  - style: formatação/código sem alteração funcional
  - refactor: refatoração
  - perf: melhoria de performance
  - test: adiciona/ajusta testes
  - chore: tarefas de build/CI/outros
- Exemplo:
  - feat(api): adicionar endpoint /api/classify
  - fix(vision): corrigir cálculo de IOU para recortes sobrepostos
- Mensagens de corpo (opcional): explique o "porquê" em linhas posteriores.
- Referencie issues: "fix: corrigir validação de upload (#42)"
- Link de referência: https://www.conventionalcommits.org/

Licença
- Este repositório pretende ser código aberto. Recomenda-se adicionar um arquivo LICENSE na raiz com MIT License (ex.: crie um arquivo `LICENSE` com o texto MIT) e incluir o cabeçalho em arquivos novos quando aplicável.

Notas finais
- Não comite o `.env` (já listado em .gitignore).
- Remova segredos de exemplo do repositório antes do push.
- Arquivos-chave:
  - [`app/services/vision.py`](app/services/vision.py)
  - [`app/agents/weee_classifier.py`](app/agents/weee_classifier.py)
  - [`app/routes.py`](app/routes.py)
  - [`requirements.txt`](requirements.txt)
  - [`.env.example`](.env.example)