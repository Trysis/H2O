# H2O
Gradient descent application on H2O molecules to optimize it's conformation.  

## Auteurs :  
Louiza GALOU  
Roude JEAN MARIE  

## À propos
Projet de M1 Bioinformatique, au sein de l'Université Paris Cité pendant l'année scolaire de 2022 à 2023.
L'objectif a été d'optimiser la conformation d'une molécule d'eau en utilisant les méthodes de descentes de gradients (fonction dérivée approximée).  

## Projet en Dynamique des macromolécules  

Pour exécuter le script et rediriger la sortie :  
`python minimisation.py file.pdb > multipdb_out.pdb`

On peut spécifier certaines valeurs :  
`python minimisation.py file.pdb [niter] [seuil] [pas] [delta] > multipdb_out.pdb  `

Puis visualiser le fichier multipdb (plusieurs modèles) :  
`vmd multipdb_out.pdb`
