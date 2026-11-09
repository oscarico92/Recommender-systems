# Plan d'action ScoutAI

Ce fichier sert de feuille de route commune. Chacun coche ses tâches au fur et à mesure (édition directe sur GitHub ou via un commit).

## Ce que le prof attend

| Semaine | Livrable | Ce qu'il vérifie |
| --- | --- | --- |
| S1 (fait) | Project Plan, 5 pages max | Clarté, faisabilité, profondeur technique |
| S2 | Status report (2 pages) + risk assessment | 5 questions oui/non (voir plus bas) |
| S3 | Status report (2 pages) + résultats | 7 questions oui/non, chaque "oui" doit pointer vers une preuve (graphe, tableau, capture, hash de commit) |
| S4 | Showcase + démo live | Pitch, démo qui marche, code montré, questions techniques |
| Final | Repo GitHub | Le prof lance le projet en moins de 10 minutes avec le README, sinon grosse perte de points |

Note finale : 50 % process (plan + status reports), 50 % produit (démo + code et doc), plus les bonus investisseurs (+1) et top 4 startups (+3).

## Règles de travail

- Une branche par personne ou par fonctionnalité (`data/...`, `model/...`, `api/...`), merge sur `main` par pull request relue par quelqu'un d'autre.
- Les données ne sont pas versionnées : un script `src/download_data.py` les récupère.
- Toute métrique sort du même script d'évaluation (`eval/`), jamais d'un notebook à part.
- Règle anti-fuite : rien de la saison de test ne sert au graphe, aux features ou au réglage des hyperparamètres.
- Point d'équipe chaque dimanche soir : ce qui est fait, ce qui bloque, qui aide qui.

## Semaine 2 (jusqu'au prochain cours) : première version

Objectif : répondre "oui" aux 5 questions du status report.

- Q1 Data : pipeline V1 construit et dataset complet traité ?
- Q2 Modèle : squelette du modèle innovant qui fait un forward pass sans planter ?
- Q3 Évaluation : harness qui renvoie un NDCG à partir de prédictions bidon ?
- Q4 Système : API qui renvoie des recommandations bidon ?
- Q5 Planning : en avance, à l'heure ou en retard ?

### Oscar (Project Lead)
- [ ] Créer l'arborescence du repo (`data/`, `src/`, `eval/`, `api/`, `app/`, `docs/`) et le `requirements.txt` de départ
- [ ] Script de split temporel par saison (train, validation, test)
- [ ] Harness d'évaluation : NDCG@10 et Recall@20, testé sur des prédictions bidon
- [ ] Baselines aléatoire et "plus populaire"
- [ ] Rédiger le status report S2

### Simon (Data Engineer)
- [ ] `src/download_data.py` : téléchargement de l'archive Transfermarkt
- [ ] Nettoyage des transferts : retirer retours de prêt, fins de carrière, transferts vers l'équipe réserve ; marquer les prêts
- [ ] Sélection d'une dizaine de championnats bien couverts et des clubs avec au moins 10 transferts
- [ ] Schéma SQLite de la table `interactions` (club, joueur, action, timestamp)
- [ ] Première version de l'endpoint `/log_interaction`

### Gautier (Lead ML Engineer)
- [ ] Construction du graphe biparti club / joueur dans PyTorch Geometric à partir des transferts d'entraînement
- [ ] Squelette LightGCN (`LGConv`), un forward pass sur le vrai graphe

### Mathis (ML Engineer Ops)
- [ ] Boucle d'entraînement BPR avec échantillonnage négatif
- [ ] Log de la loss train et validation à chaque epoch, sauvegarde des checkpoints

### Hugo (Systems Engineer)
- [ ] API FastAPI avec `/recommend` qui renvoie des recommandations bidon
- [ ] Page Streamlit qui choisit un club et appelle l'API

## Semaine 3 : intégration et premiers résultats

Objectif : répondre "oui" aux 7 questions, avec une preuve pour chacune.

- Q1 Entraînement : modèle entraîné sur tout le dataset V1, courbes qui convergent ?
- Q2 Seuil minimal : le modèle bat l'aléatoire et le "plus populaire" sur NDCG@10 et Recall@20 ?
- Q3 Fuite : évaluation sans fuite de données (split temporel) ?
- Q4 Innovation : métrique propre à l'innovation et ablation faites ?
- Q5 Système live : front, API, modèle réel, réponse en moins de 2 s, interaction visible en base ?
- Q6 Reproductibilité : quelqu'un qui n'a pas écrit le code lance tout en moins de 10 min avec le README ?
- Q7 Planning : où on en est, ce qu'il reste avant le showcase ?

### Oscar
- [ ] Tableau de résultats complet : baselines, LightGCN, cold-start, ablation
- [ ] Audit anti-fuite (dates des features, des valeurs marchandes, du graphe)
- [ ] Rédiger le status report S3 avec les preuves

### Simon
- [ ] Table des features joueurs (stats et valeurs marchandes calculées avant la date de split)
- [ ] `/log_interaction` branché sur la base, vérifié de bout en bout

### Gautier
- [ ] Encodeur cold-start (MLP features vers embeddings LightGCN) entraîné
- [ ] Évaluation sur les joueurs jamais transférés avant le split

### Mathis
- [ ] Recherche d'hyperparamètres sur la validation (nombre de couches, taille d'embedding, learning rate)
- [ ] Fonction `predict()` avec re-ranking budget et revente
- [ ] Courbes de loss train / validation

### Hugo
- [ ] Vraies prédictions de bout en bout, réponse en moins de 2 s
- [ ] Explications affichées ("similaire à X et Y recrutés en 2023") et curseur budget / revente
- [ ] README "run locally" testé par un autre membre sur une machine propre

## Semaine 4 : finalisation et showcase

- [ ] Oscar : métriques finales, graphiques (loss, ablation), 3 études de cas clubs (club vendeur, gros club, petit budget)
- [ ] Simon et Mathis : réentraînement avec les interactions loggées
- [ ] Gautier et Hugo : scénario de démo, slides de pitch, flyer d'une page
- [ ] Tous : gel du code deux jours avant le showcase, répétition du pitch

## Risques à surveiller

| Risque | Parade |
| --- | --- |
| Le "plus populaire" est dur à battre (les gros clubs achètent les mêmes joueurs) | Résultats par catégorie de club, et miser sur les métriques cold-start et budget |
| Tous les transferts ne sont pas des choix de recrutement | Règles de filtrage documentées dès la S2 |
| Couverture inégale des petits championnats | Se limiter à une dizaine de championnats en V1 |
| Fuite via les valeurs marchandes ou les stats de saison | Toutes les features calculées avec une date de coupure, audit en S3 |
| Le dataset n'est plus mis à jour (snapshot juillet 2026) | On travaille sur le snapshot, les interactions loggées apportent le neuf |
