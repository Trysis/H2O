# H2O
Gradient descent application on H2O molecules to optimize it's conformation.  

### Auteurs :  
Louiza GALOU  
Roude JEAN MARIE  

---
### Projet en Dynamique des macromolécules  

Pour exécuter le script et rediriger la sortie :  
`python minimisation.py file.pdb > multipdb_out.pdb`

On peut spécifier certaines valeurs :  
`python minimisation.py file.pdb [niter] [seuil] [pas] [delta] > multipdb_out.pdb  `

Puis visualiser le fichier multipdb (plusieurs modèles) :  
`vmd multipdb_out.pdb`
