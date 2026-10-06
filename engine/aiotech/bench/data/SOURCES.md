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
