# AIOTECH Search

Moteur de recherche cognitif auto-hébergé. Il répond à une seule question : **quel est le chemin le plus
fiable et vérifié de la question à la réponse ?** Chaque requête est lue de plusieurs façons, chaque lecture
suit sa trajectoire de preuves, et le résultat se réduit à deux ou trois réponses classées par fiabilité, avec
le statut de chaque affirmation : `[FAIT]`, `[INFÉRENCE]`, `[INCERTAIN]` ou `[NON VÉRIFIÉ]`.

- Interface 3D : barre de recherche en verre, nuage de neurones (les interprétations de la requête) reliés par
  des synapses, qui se contracte vers les réponses. Les clics ne départagent que des réponses de fiabilité égale.
- API pour tous les LLM : Claude par défaut, tout autre modèle via LiteLLM ; compatible OpenAI, outils, MCP.
- Fonctionne sans clé : hors ligne, les réponses sont produites par règles et gabarits vérifiés, à 0 jeton.

## Démarrage

```bash
cp .env.example .env          # facultatif : ANTHROPIC_API_KEY, clés d'API publiques
docker compose up -d          # moteur, interface 3D, SearXNG, Redis
```

Ouvrir http://localhost:8080. Seul nginx publie un port ; le moteur, Redis et SearXNG restent sur un réseau
interne. Les secrets (clé de l'interface web, secret SearXNG) sont générés au premier démarrage.

Derrière un proxy qui intercepte TLS, fournir son certificat au build :
`docker build --secret id=build_ca,src=/chemin/ca.pem …` (le secret est facultatif et n'entre dans aucune image).

## Benchmark en une commande

```bash
docker compose run --rm bench
```

CPU uniquement, sans réseau ni clé, environ 35 secondes sur 4 cœurs. Le rapport s'affiche en Markdown et
s'écrit dans `./bench-results/` (si le dossier n'est pas accessible en écriture pour le conteneur :
`docker compose run --rm --user "$(id -u):$(id -g)" bench`). Avec `--llm` et une clé, un même lecteur LLM
ajoute EM/F1 (HotpotQA) et l'exactitude des labels (FEVER) pour tous les systèmes.

Mesures du 2026-10-07 (4 cœurs, Python 3.13, sans GPU), contre une RAG classique (BM25 top-k) sur les mêmes
documents ; entre crochets, l'intervalle de confiance à 95 % de l'écart (bootstrap apparié, 2 000 tirages) :

| Mesure | RAG classique | AIOTECH Search | Écart |
|---|---|---|---|
| HotpotQA (n = 300) : les 2 paragraphes utiles en tête (top 2) | 42,0 % | 59,7 % | +17,7 [11,7 ; 24,0] |
| HotpotQA : paragraphes utiles retrouvés (top 2) | 69,0 % | 78,3 % | +9,3 [6,2 ; 12,7] |
| HotpotQA : réponse attendue dans la réponse de tête, sans LLM (n = 286) | 26,6 % | 26,9 % | +0,4 [−4,9 ; 5,2] |
| FEVER (n = 355) : preuve exacte en 1re position | 39,1 % | 39,4 % | +0,3 [−0,9 ; 1,4] |
| FEVER : réponse de tête appuyée sur la preuve exacte | 39,1 % | 42,2 % | +3,1 [0,3 ; 5,9] |
| Latence p50 / p95 par requête (HotpotQA, pipeline complet) | 0,2 / 0,3 ms | 12,2 / 22,1 ms | |
| VRAM, jetons LLM, coût API | 0 | 0 | |

Le gain sur HotpotQA vient de la recherche (saut par les liens de titre entre documents) ; sans LLM, la
réponse extraite n'est pas meilleure que la phrase de tête de BM25. L'énergie est une estimation (temps CPU ×
15 W par cœur, ou RAPL si lisible) ; les FLOPs d'un LLM distant ne sont pas mesurables côté client et ne sont
pas publiés. Échantillons, sources, licences et séparation réglage / test :
[`engine/aiotech/bench/data/SOURCES.md`](engine/aiotech/bench/data/SOURCES.md).

## Architecture

```
navigateur ──► nginx (seul port publié, CSP stricte, quotas)
                 ├─ /            interface 3D (Vite, Three.js)
                 ├─ /api/ui/…    flux SSE de l'interface (clé web injectée par nginx)
                 ├─ /v1/…, /mcp  API publique (clé d'API obligatoire)
                 └─ /admin       réseau local seulement, clé + jeton
               moteur FastAPI ──► Redis (cache, quotas, clics)   [réseau interne]
                              ──► SearXNG (web)                  [réseau interne + sortie]
                              ──► Neo4j (facultatif, profil graph)
                              ──► LLM : Claude (Anthropic) ou LiteLLM
```

Étapes d'une recherche (`engine/aiotech/pipeline.py`), diffusées en direct à l'interface :

1. **Garde-fous** : masquage RGPD (cartes bancaires par contrôle de Luhn, IBAN, IP, téléphones, e-mails)
   avant tout traitement ; injection de prompt (FR/EN) bloquée dès l'entrée, à 0 jeton payé.
2. **Intention** : type de question, contraintes chiffrées, entités, interprétations T1…Tn.
   L'`AdaptiveComputeGate` dose trajectoires, passages, sauts et appels LLM selon la difficulté.
3. **Recherche hybride** : BM25 et vecteurs fusionnés par scores normalisés, sur le corpus et le web
   (SearXNG), puis saut par les liens de titre entre documents ; second saut par entités pont au palier
   approfondi.
4. **Graphe** : entités, valeurs et sources ; les contradictions (A dit X, B dit Y, X ≠ Y) sont détectées
   mécaniquement puis tranchées par la fiabilité des sources, ou laissées incertaines.
5. **Raisonnement SCG puis TAP** : t-norme de Gödel `min(a, b)` ; une prémisse fausse ou une contrainte
   violée rejette la trajectoire, et la fiabilité ne dépasse jamais la prémisse la plus faible.
6. **Vérification et réponses** : statut de chaque affirmation, 2 ou 3 possibilités classées par fiabilité,
   puis par clics à égalité ; rédaction par gabarit, ou par LLM sous contrôle des nombres cités.

## API

Chaque appel exige une clé (`AIOTECH_API_KEYS=nom:clé`, 16 caractères minimum) et respecte les quotas.

```bash
curl -N -H "x-api-key: $AIOTECH_KEY" "http://localhost:8080/v1/search/stream?q=délai+de+rétractation"
curl -H "x-api-key: $AIOTECH_KEY" -H 'content-type: application/json' \
     -d '{"query": "Quel PC à 500 € pour AIOTECH 44 ?"}' http://localhost:8080/v1/search
```

- Compatible OpenAI : `POST /v1/chat/completions` (avec ou sans `stream`), `GET /v1/models`, modèle
  `aiotech-search`. N'importe quel client OpenAI s'y branche avec `base_url=http://…/v1`.
- Outils : `GET /v1/tools` (schémas Anthropic, ou OpenAI avec `?format=openai`), `POST /v1/tools/{nom}` : `aiotech_search`,
  `calculate` (calculatrice statique, sans exécution de code) et `read_file` (lecture confinée au dossier
  monté en `/data/docs`, jamais au-delà ; le `.env` et les fichiers système sont hors d'atteinte).
- MCP (HTTP) : `http://…/mcp/` avec l'en-tête `x-api-key`.
- Administration : `/admin/stats`, `POST /admin/documents`, `DELETE /admin/documents/{id}`, avec une clé et
  l'en-tête `x-admin-token`, depuis le réseau local seulement. Sans `AIOTECH_ADMIN_TOKEN`, l'administration
  est désactivée ; la clé de l'interface web n'y donne jamais accès.

## Configuration

Toutes les variables sont décrites dans [`.env.example`](.env.example). Les principales :
`AIOTECH_LLM_PROVIDER` (`anthropic`, `litellm` ou `none`), `AIOTECH_LLM_MODEL`, `ANTHROPIC_API_KEY`,
`AIOTECH_API_KEYS`, `AIOTECH_ADMIN_TOKEN`, `AIOTECH_QUOTA_PER_MINUTE` et `AIOTECH_QUOTA_PER_DAY`.
Neo4j s'ajoute avec `docker compose --profile graph up -d` et `AIOTECH_NEO4J_URL` dans `.env`.

## Développement

```bash
cd engine && python -m venv .venv && .venv/bin/pip install -e ".[dev,mcp,providers]"
.venv/bin/ruff check . && .venv/bin/mypy aiotech && .venv/bin/pytest -q
cd ../web && npm ci && npm run dev        # proxy vers le moteur sur :8000 (AIOTECH_WEB_API_KEY)
```

La CI (`.github/workflows/ci.yml`) rejoue ces contrôles, construit les images, exécute les tests dans
l'image sans réseau, lance le benchmark en une commande et vérifie les garde-fous à travers nginx.
