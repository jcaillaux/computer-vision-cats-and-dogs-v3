# DOCKERFILE - Application FastAPI MLOps

```dockerfile
# ═══════════════════════════════════════════════════════════════════════════════
# 🐳 DOCKERFILE - Application FastAPI MLOps
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🎯 OBJECTIF
# Conteneuriser l'application FastAPI avec toutes ses dépendances pour garantir
# un environnement reproductible (dev = prod). Optimisé pour production avec
# image légère, layers en cache, et healthcheck intégré.
#
# 📚 CONCEPTS CLÉS
# - Multi-stage builds : non utilisé ici (app simple), mais à considérer si >500MB
# - Layer caching : ordre COPY optimisé (dépendances avant code)
# - Image slim : -60% vs image standard Python
# - Healthcheck : vérification automatique de l'état du conteneur
#
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim
# 📦 IMAGE DE BASE : Python 3.11 version "slim"
# 
# POURQUOI 3.11 ?
# - Performance : +25% vs 3.10 (PEP 659 - specialized adaptive interpreter)
# - Compatibilité : TensorFlow 2.15+, FastAPI 0.100+
# - Support LTS : sécurité garantie jusqu'en 2027
#
# POURQUOI SLIM ?
# - Taille : ~120MB (vs ~900MB pour python:3.11 standard)
# - Debian-based : compatible avec apt-get (vs Alpine qui utilise apk)
# - Compromis : librairies de base présentes, pas de bloat inutile
# 
# ALTERNATIVES
# - python:3.11-alpine : ultra-léger (~50MB) mais compilations complexes (TensorFlow)
# - python:3.11 : toutes les libs système, utile pour debug mais lourd en prod

WORKDIR /app
# 📁 RÉPERTOIRE DE TRAVAIL
# Tous les COPY et RUN suivants s'exécutent depuis /app
# Équivalent à : RUN mkdir -p /app && cd /app

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 INSTALLATION DÉPENDANCES SYSTÈME
# ═══════════════════════════════════════════════════════════════════════════════
RUN apt-get update && apt-get install -y \
    curl \
    # 🩺 curl : nécessaire pour HEALTHCHECK (test endpoint /health)
    # Alternative : wget, mais curl plus standard pour API testing
    && rm -rf /var/lib/apt/lists/*
    # 🧹 NETTOYAGE : supprime cache apt (~100MB économisés)
    # /var/lib/apt/lists/ contient les métadonnées des packages
    # Bonne pratique : TOUJOURS nettoyer dans la même layer (optimisation taille)

# 💡 OPTIMISATION LAYER CACHING
# apt-get update && install && rm dans un SEUL RUN :
# ✅ 1 layer au lieu de 3 → image plus petite
# ✅ Cache invalide si dépendances changent → rebuild propre

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 INSTALLATION DÉPENDANCES PYTHON
# ═══════════════════════════════════════════════════════════════════════════════
COPY requirements/base.txt requirements/prod.txt requirements/monitoring.txt ./
# 📋 COPIE SÉPARÉE DES REQUIREMENTS (avant le code source)
# 
# POURQUOI CETTE ORDRE ?
# Docker met en cache chaque layer. Si requirements changent → rebuild.
# Mais si SEUL le code change → réutilise cache pip (gain de temps énorme)
#
# STRATÉGIE DE REQUIREMENTS
# - base.txt : dépendances core (FastAPI, TensorFlow, SQLAlchemy)
# - prod.txt : outils production (gunicorn, uvicorn workers)
# - monitoring.txt : Prometheus client, psutil
# Séparation = clarté + réutilisabilité (ex: base.txt partagé avec notebooks)

#RUN pip install --no-cache-dir \
#    -r base.txt \
#    -r prod.txt \
#    -r monitoring.txt
RUN pip install --no-cache-dir uv
RUN uv add --requirements monitoring.txt
# 🐍 INSTALLATION AVEC PIP
#
# --no-cache-dir : ne stocke PAS les wheels téléchargés (~300MB économisés)
# En prod, pas besoin de cache (build une fois, run partout)
#
# ORDRE D'INSTALLATION
# 1. base.txt (dépendances lourdes : TensorFlow ~400MB)
# 2. prod.txt (léger : gunicorn, uvloop)
# 3. monitoring.txt (léger : prometheus-client)
# → Si monitoring change, pas de retéléchargement de TensorFlow (cache layer)

# 💡 ALTERNATIVE POUR TRÈS GROSSES IMAGES
# Multi-stage build (non nécessaire ici) :
#   FROM python:3.11-slim AS builder
#   RUN pip install --user ...
#   FROM python:3.11-slim
#   COPY --from=builder /root/.local /root/.local
# Permet de séparer outils de build vs runtime (~30% gain supplémentaire)

# ═══════════════════════════════════════════════════════════════════════════════
# 📂 COPIE DU CODE SOURCE
# ═══════════════════════════════════════════════════════════════════════════════
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
# 📁 COPIE SÉLECTIVE (pas de COPY . .)
#
# STRUCTURE PRÉSERVÉE DE LA V2
# /app/
#   ├── src/        → code métier (api/, models/, monitoring/)
#   ├── scripts/    → run_api.py (entrypoint)
#   └── config/     → settings.py
#
# POURQUOI PAS "COPY . ." ?
# ❌ Copierait aussi : .git/, tests/, notebooks/, __pycache__/, .env
# ✅ Copie sélective = image propre + sécurité (pas de secrets accidentels)
# 
# ORDRE STRATÉGIQUE
# Copié EN DERNIER = invalidation cache uniquement si code change
# Si requirements inchangés → build ultra-rapide (réutilise layers pip)

# ⚠️ FICHIERS NON COPIÉS (gérés par volumes Docker Compose)
# - data/ : dataset monté en read-only (../data:/app/data:ro)
# - models/ : fichier .h5 monté en read-only (../models:/app/models:ro)
# Avantage : update modèle sans rebuild image (juste restart conteneur)

# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 EXPOSITION DU PORT
# ═══════════════════════════════════════════════════════════════════════════════
EXPOSE 8000
# 📡 DOCUMENTATION DU PORT (métadonnée uniquement)
#
# ⚠️ EXPOSE NE PUBLIE PAS LE PORT !
# C'est docker run -p 8000:8000 ou docker-compose ports: qui le fait
# Rôle : documenter l'intention (quel port l'app utilise)
#
# Standard FastAPI : 8000 (convention, modifiable via uvicorn --port)

# ═══════════════════════════════════════════════════════════════════════════════
# 🏥 HEALTHCHECK - Surveillance automatique
# ═══════════════════════════════════════════════════════════════════════════════
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# 🩺 VÉRIFICATION PÉRIODIQUE DE L'ÉTAT DU CONTENEUR
#
# PARAMÈTRES
# --interval=30s : fréquence des checks (toutes les 30s après start-period)
# --timeout=10s : durée max d'un check (inférence CNN peut être lente)
# --start-period=40s : grace period (permet chargement modèle TensorFlow)
# --retries=3 : nombre d'échecs avant status "unhealthy"
#
# COMMANDE DE TEST
# curl -f http://localhost:8000/health
#   -f : fail (exit code ≠ 0) si HTTP status ≠ 2xx/3xx
#   || exit 1 : force exit code 1 si curl échoue
#
# ENDPOINT /health REQUIS (à implémenter dans FastAPI)
# Exemple de réponse :
#   {
#     "status": "healthy",
#     "database": "connected",
#     "model": "loaded",
#     "timestamp": "2025-11-16T10:30:00Z"
#   }
#
# ÉTATS RÉSULTANTS
# - starting : pendant start-period (40s)
# - healthy : check réussi
# - unhealthy : 3 échecs consécutifs (3×30s = 90s)
#
# 💡 UTILISATION PAR DOCKER COMPOSE
# depends_on avec condition: service_healthy attend ce healthcheck

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 COMMANDE DE DÉMARRAGE
# ═══════════════════════════════════════════════════════════════════════════════
CMD ["python", "scripts/run_api.py"]
# 🏃 ENTRYPOINT DE L'APPLICATION
#
# POURQUOI run_api.py (et pas directement uvicorn) ?
# ✅ Cohérence avec V2 (même point d'entrée)
# ✅ Flexibilité : peut inclure setup pré-démarrage (logging, warmup modèle)
# ✅ Configuration centralisée (workers, host, port dans le script)
#
# CONTENU TYPIQUE DE run_api.py :
#   import uvicorn
#   if __name__ == "__main__":
#       uvicorn.run(
#           "src.api.main:app",
#           host="0.0.0.0",      # écoute sur toutes interfaces (requis Docker)
#           port=8000,
#           workers=4,           # multi-processing (production)
#           log_level="info"
#       )
#
# FORMAT EXEC vs SHELL
# ["python", "..."] = exec form (RECOMMANDÉ)
#   ✅ PID 1 = python (gestion signaux propre : SIGTERM → graceful shutdown)
#   ✅ Pas de shell intermédiaire
# "python ..." = shell form
#   ❌ PID 1 = /bin/sh (ne transmet pas SIGTERM correctement)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎓 CONCEPTS AVANCÉS (non implémentés ici, pour aller plus loin)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. UTILISATEUR NON-ROOT (sécurité)
#    RUN useradd -m appuser
#    USER appuser
#    → Évite exécution en root (principe du moindre privilège)
#
# 2. MULTI-STAGE BUILD (optimisation taille)
#    FROM python:3.11-slim AS builder
#    RUN pip install --user -r requirements.txt
#    FROM python:3.11-slim
#    COPY --from=builder /root/.local /root/.local
#    → Sépare build vs runtime (supprime gcc, headers, etc.)
#
# 3. LABELS (métadonnées)
#    LABEL maintainer="remi@example.com"
#    LABEL version="3.0.0"
#    LABEL description="CV Cats/Dogs MLOps API"
#    → Traçabilité (docker inspect montre les labels)
#
# 4. ARG POUR VERSIONS DYNAMIQUES
#    ARG PYTHON_VERSION=3.11
#    FROM python:${PYTHON_VERSION}-slim
#    → Build avec différentes versions Python
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🛠️ COMMANDES BUILD & DEBUG
# ═══════════════════════════════════════════════════════════════════════════════
#
# BUILD
#   docker build -t cv-app:v3 -f docker/Dockerfile.app .
#   docker build --no-cache ...  # force rebuild sans cache
#
# INSPECTION
#   docker history cv-app:v3     # voir taille de chaque layer
#   docker inspect cv-app:v3     # métadonnées complètes
#
# DEBUG
#   docker run -it cv-app:v3 bash           # shell interactif
#   docker run cv-app:v3 python --version   # override CMD
#
# OPTIMISATION
#   docker images | grep cv-app              # vérifier taille finale
#   dive cv-app:v3                           # analyser layers (outil externe)
#
# ═══════════════════════════════════════════════════════════════════════════════
```