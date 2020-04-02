# Use case pour BSdgital

## Requirement

```bash
pip install -r requirements.txt
```

## Structure du projet

### utils.py

Dans ce fichier il y a toutes les fonctions qui permettent de créer les données d'entrainement et de validation à partir des images droites

### correct_rotation.py

"main" du projet, c'est là qu'on fait le lien entre tous les elements du projet

### train/train_facture.py

Dans ce fichier se trouve les fonctions qui créent le modele


### data/facture.py

Dans ce fichier on retrouve la fonction qui permet de recuperer la liste des path des données d'entrainement et la liste des path des données de test

## Entrainer le modele

```bash
python3 <path_to_git_folder>/BSDigital/train/facture.py <path_to_train_data>
```

## utiliser le modele

```bash
python <path_to_git_folder>/BSDigital/correct_rotation.py <path_to_train_model> <path_to_real_data>
```

## Auteurs

Luc Antimi
Arthur Aries
Grégoire Effroy
Léna Sasal
