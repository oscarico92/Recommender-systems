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
- [x] Créer l'arborescence du repo (`data/`, `src/`, `eval/`, `api/`, `app/`, `docs/`) et le `requirements.txt` de départ
- [x] Script de split temporel par saison (train, validation, test) : `src/split.py`
- [x] Harness d'évaluation : NDCG@10 et Recall@20, testé : `eval/harness.py`
- [x] Baselines aléatoire et "plus populaire" (plus "plus grosse valeur marchande") : `eval/baselines.py`
- [x] Rédiger le status report S2 (rendu le 16/11)

### Simon (Data Engineer)
- [x] `src/download_data.py` : téléchargement de l'archive Transfermarkt
- [x] Nettoyage des transferts : retirer retours de prêt, fins de carrière, transferts vers l'équipe réserve ; marquer les prêts (`src/clean_transfers.py`, chiffres dans `data/README.md`)
- [x] Sélection d'une dizaine de championnats bien couverts et des clubs avec au moins 10 transferts : 26 369 interactions, 322 clubs, 9 877 joueurs
- [x] Schéma SQLite de la table `interactions` (club, joueur, action, timestamp) : `src/db.py`
- [x] Première version de l'endpoint `/log_interaction` : `api/interactions.py`, testée

### Gautier (Lead ML Engineer)
- [x] Construction du graphe biparti club / joueur dans PyTorch Geometric à partir des transferts d'entraînement : `src/graph.py`
- [x] LightGCN (PyG) entraîné sur le vrai graphe : `src/models/lightgcn.py`

### Mathis (ML Engineer Ops)
- [x] Boucle d'entraînement BPR avec échantillonnage négatif : `src/train.py`
- [x] Log de la loss train et validation, early stopping, modèle exporté dans `models/serving/`

### Hugo (Systems Engineer)
- [x] API FastAPI avec `/recommend` (vraies recommandations) et `/clubs` : `api/main.py`
- [x] Page Streamlit qui choisit un club et appelle l'API : `app/streamlit_app.py`

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
- [x] Tableau de résultats complet : baselines, LightGCN, cold-start, ablation : `eval/results/test_results.md`
- [x] Audit anti-fuite (dates des features, des valeurs marchandes, du graphe) : `docs/week3_evidence.md`, `tests/test_features_leakage.py`
- [x] Revue de code de la PR #1 (deux tours de review), merge dans `main` (`a40c704`)
- [x] Rédiger le status report S3 avec les preuves (base : `docs/week3_evidence.md`)

### Simon
- [x] Table des features joueurs (stats et valeurs marchandes calculées avant la date de split) : `src/features.py`
- [x] `/log_interaction` branché sur la base, vérifié de bout en bout (clic Streamlit vers SQLite)

### Gautier
- [x] Encodeur cold-start (MLP features vers embeddings LightGCN) entraîné : `src/models/coldstart.py`
- [x] Évaluation sur les joueurs inconnus du graphe avant le split (liste "nouveaux talents")

### Mathis
- [x] Recherche d'hyperparamètres sur la validation : `src/tune.py`, `eval/results/tuning.csv`
- [x] Fonction `predict()` avec re-ranking budget et revente : `src/predict.py`, `src/rerank.py`
- [x] Courbes de loss train / validation : `docs/figures/loss_curves.png`

### Hugo
- [x] Vraies prédictions de bout en bout, réponse en moins de 2 s (environ 20 ms)
- [x] Explications affichées ("similaire à X et Y recrutés en 2023") et curseur budget / revente
- [x] README "run locally" testé par un autre membre sur une machine propre : Oscar, le 17/11, sous Windows, dossier vide, moins de 5 min (commit `638cf4a`)

## Semaine 4 : finalisation et showcase

Priorités issues du test de la semaine 3 (voir les issues GitHub).

### Gautier et Mathis
- [x] Biais d'âge : terme `age_weight` dans le re-ranking (`src/rerank.py`), branché dans l'API, Streamlit ("Youth focus") et `eval/run_experiments.py`. Sur le modèle servi, top 10 de 29,3 ans à 27,9 (w=0,5) et 26,7 (w=1)
- [x] Biais de popularité : terme `mobility_weight` (pénalité sur log(1 + nombre de transferts passés)), même branchement
- [x] Poids choisis sur la validation avec 3 graines et une grille d'âge allant jusqu'à 2 : revente 1, âge 2, mobilité 0 (`eval/results/tuning_rerank.csv`, `best_rerank.json`). La pénalité de mobilité a été testée, mais n'est pas retenue par la règle de sélection. Évaluation test dans `eval/results/test_results.md` ; ces poids sont exportés comme valeurs par défaut de l'API. Voir aussi `eval/results/seed_stability.md` pour l'instabilité entre graines.

### Hugo et Gautier
- [x] Explications des nouveaux talents : les embeddings cold-start sont quasi colinéaires (6 dimensions portent 80 % de la variance, contre 25 pour LightGCN), donc le cosinus renvoyait toujours les mêmes recrues. Les nouveaux talents sont désormais expliqués par une règle métier de profil, et non par la raison du classement du modèle (même sous-poste, puis même poste, puis âge à la signature le plus proche) : paires distinctes dans le top 10 de 38 % à 72 %, même poste de 23 % à 90 %. Test : `tests/test_explanations.py`
- [x] Liste principale (issue #7) : les recrues du même poste passent en premier (toujours classées par similarité d'embedding), complétées par les autres si le club en a moins de 2. Vérification avec les réglages finaux (revente 1, âge 2, mobilité 0) : explications du même poste de 24,35 % à 95,05 %, paires distinctes de 74,19 % à 70 %. Classements inchangés sur les 322 clubs. Test : `tests/test_explanations.py`

### Hugo : vérification du système
- [x] Après Shortlist ou Dismiss, recharger les deux listes pour appliquer le feedback immédiatement dans l'interface
- [x] Signaler les joueurs hors budget et distinguer les comparaisons de profils des similarités de recrutement
- [x] Gérer les erreurs API dans la page, sans confondre un historique indisponible avec un historique vide
- [x] Tester le parcours page / API / SQLite avec une base temporaire : `tests/test_streamlit_app.py`, `tests/test_recommend_api.py`
- [x] Préparer les étapes de démo et les notes de prise en main : `docs/systems_engineer_notes.md`
- [ ] Relecture personnelle des changements et répétition de la partie système avant le showcase

### Simon et Mathis
- [x] Feedback de l'app : mise à jour immédiate du vecteur du club dans l'API (Rocchio), réentraînement possible avec `python -m src.export_model --feedback`, vérification dans `eval/results/feedback_check.md`
- [x] Pondération des prêts testée (aucune preuve de gain, poids gardé à 1) : `eval/results/loan_weight.md`

### Oscar
- [ ] Métriques finales et graphiques (loss, ablation) après les corrections, mise à jour de `eval/results/`
- [ ] 3 études de cas clubs (club vendeur, gros club, petit budget) pour le pitch
- [ ] Vérifier que le README tourne toujours sur une machine propre après les derniers merges

### Tous
- [ ] Scénario de démo, slides de pitch, flyer d'une page
- [ ] Chacun relit et sait expliquer sa partie du code (le prof posera des questions techniques au showcase)
- [ ] Gel du code deux jours avant le showcase, répétition du pitch

## Risques à surveiller

| Risque | Parade |
| --- | --- |
| Le "plus populaire" est dur à battre (les gros clubs achètent les mêmes joueurs) | Résultats par catégorie de club, et miser sur les métriques cold-start et budget |
| Tous les transferts ne sont pas des choix de recrutement | Règles de filtrage documentées dès la S2 |
| Couverture inégale des petits championnats | Se limiter à une dizaine de championnats en V1 |
| Fuite via les valeurs marchandes ou les stats de saison | Toutes les features calculées avec une date de coupure, audit en S3 |
| Le dataset n'est plus mis à jour (snapshot juillet 2026) | On travaille sur le snapshot, les interactions loggées apportent le neuf |
