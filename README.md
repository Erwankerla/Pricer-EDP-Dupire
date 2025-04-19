# Modèle de Dupire & EDP

Ce projet explore la valorisation d’un call spread sur le CAC 40 à l’aide des **équations aux dérivées partielles (EDP)**, dans un cadre de **volatilité locale selon le modèle de Dupire**.

📊 Les **résultats** du pricer sur différentes dates, comparées à celles obtenues via SuperDerivatives, sont disponibles dans le dossier `Inputs-Outputs/`, notamment dans le fichier `TestPrixSuperD.xlsx`.

📄 Un document complet décrivant toutes les étapes de modélisation, de calibration et de pricing est disponible dans le dossier `Documents/`.

📚 Tous les articles théoriques utilisés pour construire la méthode sont référencés en fin de note, et les formules implémentées sont issues de sources reconnues (Gatheral, Rouah, Tankov) dans `Articles/`.

💻 Le code Python associé est structuré en plusieurs modules, chacun correspondant à une étape clé du projet dans `Implémentation/`:
- `DataFetcher.py` : récupération et formatage des données de volatilité implicite
- `SVI.py` : calibration de la surface de volatilité implicite via le modèle SSVI
- `LocalSurface.py` : construction de la surface de volatilité locale à partir des smiles SVI
- `PDE.py` : résolution de l’EDP de Dupire via des schémas numériques (Explicite, Implicite, Crank-Nicholson)
- `main.py` : script principal permettant d’enchaîner toutes les étapes et de valoriser le call spread

---
**Objectif** : proposer un pricer robuste, fondé sur la théorie moderne des dérivées, permettant de valoriser des options dans un cadre réaliste de volatilité locale.
---
