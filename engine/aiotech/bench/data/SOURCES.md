# Échantillons figés du benchmark

Les deux fichiers sont produits par `python -m aiotech.bench.prepare` (déterministe : relancer la commande
redonne exactement les mêmes octets) et versionnés pour que le benchmark tourne sans réseau.

## hotpotqa_dev_distractor_s300.jsonl.gz

- Source : `hotpot_dev_distractor_v1.json`, dev « distractor » officiel de HotpotQA (Yang et al., EMNLP 2018),
  7 405 questions, 46 320 117 octets,
  SHA-256 `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`
  (copie identique au fichier de http://curtis.ml.cmu.edu/datasets/hotpot/, récupérée via le dépôt public
  nju-websoft/KG2RAG ; l'empreinte est vérifiée par le script).
- Tirage : 300 questions sans remise, `random.Random(20261006).sample`, conservées dans l'ordre du fichier.
- Champs gardés : identifiant, question, réponse, type, faits support, 10 paragraphes de contexte.
- Licence : CC BY-SA 4.0 (HotpotQA).
- SHA-256 du fichier versionné : `5f9ad390c9c0e72382226375113785d07bdf4e06df7c4029adb5c4e5c71aaa21`

## fever_dev_pairs.jsonl.gz

- Source : FeverSymmetric v0.2 (Schuster et al., EMNLP 2019), fichiers `fever_symmetric_dev.jsonl` et
  `fever_symmetric_test.jsonl`, commit `fa25d17e73ea0694edc78133ec654b9f32f6db73` de
  github.com/TalSchuster/FeverSymmetric.
- Affirmations : les 355 paires originales du dev FEVER (Thorne et al., NAACL 2018), identifiants FEVER sans
  suffixe `000000N` : 147 SUPPORTS, 208 REFUTES, chacune avec sa phrase de preuve Wikipédia.
- Corpus de preuves : 648 phrases distinctes, soit les preuves originales et les phrases modifiées de
  FeverSymmetric, gardées comme leurres difficiles (jamais comme preuve attendue).
- Licence : CC BY-SA 3.0 (Wikipédia), voir DATA_LICENSE de FeverSymmetric.
- SHA-256 du fichier versionné : `b284969ffc176a1c447c263c253ff1a6f351aa06f10063e7a3df9aa103191dae`

## Limites à connaître

- FEVER complet demande le vidage Wikipédia de 2017 ; ici la recherche se fait dans un corpus commun de 648
  phrases, ce qui mesure le classement des preuves et non la recherche dans tout Wikipédia.
- HotpotQA « distractor » fournit 10 paragraphes par question : on mesure la sélection des paragraphes utiles
  et de la phrase de réponse, pas la recherche ouverte (« fullwiki »).

## Réglage de la recherche, séparé du test

Les réglages `VECTOR_WEIGHT` (retrieval/index.py) et `RetrievalParams` (pipeline.py) ne sont jamais choisis
sur les 300 questions du benchmark. Ils viennent de `python -m aiotech.bench.tune`, qui tire ses questions
parmi les 7 105 questions du même dev « distractor » absentes de l'échantillon de test (exclusion vérifiée).

Historique, par transparence : la première version (fusion par rangs réciproques, vecteurs au même poids
que BM25, sans saut par liens) a été mesurée sur l'échantillon de test le 2026-10-07 et faisait moins bien
que BM25 seul (rappel @2 : 64,7 % contre 69,0 %). C'est ce résultat qui a déclenché le diagnostic ; le
diagnostic et le réglage ont ensuite été faits uniquement hors test :

- 1 000 questions hors test (`--seed 7`), recherche seule, 10 premiers documents :

| Réglage | r@2 | r@5 | tous@2 | tous@5 |
|---|---|---|---|---|
| BM25 seul (RAG classique) | 65,7 | 85,6 | 36,5 | 72,2 |
| première version (vecteurs 1,0, entités 1,0, pas de liens) | 58,8 | 77,8 | 27,4 | 58,3 |
| vecteurs 0,1, entités 0,5, pas de liens | 64,7 | 84,2 | 34,9 | 69,5 |
| **retenu : vecteurs 0,1, entités 0,5, liens 0,5** | **74,2** | **92,8** | **51,7** | **86,2** |
| vecteurs 0,0, entités 0,5, liens 0,5 | 74,5 | 92,7 | 52,1 | 86,0 |
| vecteurs 0,1, entités 0,5, liens 1,0 | 72,0 | 93,0 | 48,7 | 86,4 |

  La grille complète (24 réglages) s'obtient avec la commande ci-dessus. Les vecteurs hachés sont un signal
  lexical : sur cet anglais ils n'apportent rien, mais ils rattrapent les variantes d'écriture (pluriels,
  accents, fautes) en français ; un poids de 0,1 en garde l'effet pour un coût de 0,3 point ici.
- Contrôle sur un second tirage hors test (`--seed 11`, 1 000 questions) avec les réglages retenus : rappel @2
  74,7 % (BM25 66,5 %), tous@2 53,2 % (BM25 37,3 %).
- FEVER : contrôle sur les 1 065 affirmations générées de FeverSymmetric (jamais dans le test) : preuve
  exacte en 1re position 43,2 % pour BM25 et 43,2 à 43,3 % pour AIOTECH selon le poids des vecteurs
  (0, 0,1, 0,25) ; le réglage ne change donc presque rien sur FEVER, où il n'y a pas de liens de titre.
