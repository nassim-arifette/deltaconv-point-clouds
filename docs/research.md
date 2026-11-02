# DeltaConv : audit de l'indépendance au choix des bases

Vérifié le 1 novembre 2026. Le dossier historique contenait des notes de présentation et leur PDF, sans implémentation ni résultats expérimentaux enregistrés. Le présent travail ajoute un audit numérique autonome et une voie distincte vers le code officiel.

## Question et sources

[DeltaConv, Wiersma et al., SIGGRAPH / TOG 2022](https://doi.org/10.1145/3528223.3530166) construit des convolutions anisotropes à partir de champs scalaires, vectoriels et d'opérateurs différentiels. Le [texte final, §3.2](https://arxiv.org/html/2111.08799v5) souligne l'instabilité des moindres carrés dans les voisinages mal échantillonnés et ajoute une régularisation. Le §3.7 discute l'indépendance au choix des bases tangentielles. Ces deux propriétés motivent notre question : **les choix numériques de régularisation et de normalisation préservent-ils cette indépendance ?** Il s'agit d'une question sur la discrétisation, pas d'une évaluation des performances publiées.

Le [code officiel](https://github.com/rubenwiersma/deltaconv) est fixé au [commit `186fec369fa2ceb4559830bc421282dddb2300a2`](https://github.com/rubenwiersma/deltaconv/commit/186fec369fa2ceb4559830bc421282dddb2300a2), vérifié avec l'API GitHub. Dans [`grad_div_mls.py`](https://github.com/rubenwiersma/deltaconv/blob/186fec369fa2ceb4559830bc421282dddb2300a2/deltaconv/geometry/grad_div_mls.py), la base polynomiale est `[1,u,v,u²,uv,v²]`, les poids sont gaussiens avec largeur moyenne par forme, et la régularisation ajoute `λ I`. Le normaliseur regroupe les sommes des valeurs absolues des deux composantes tangentielles. Ces formules sont transcrites en NumPy pour l'audit ; divergence et réseau ne sont pas réimplémentés.

Les [tests amont](https://github.com/rubenwiersma/deltaconv/blob/186fec369fa2ceb4559830bc421282dddb2300a2/test/geometry/test_grad_div_mls.py) vérifient déjà plusieurs propriétés géométriques. Leur contrôle de changement de base emploie notamment `regularizer=1e-8` et une tolérance. Notre expérience porte sur `λ=1e-3`, utilisé par défaut dans le modèle, et distingue les effets des deux opérations numériques.

## Pourquoi la pénalité dépend de la base

Écrivons la partie quadratique comme `q(u,v)=a u²+b uv+c v²`. Le tenseur symétrique associé est

```text
H = [[a, b/2],
     [b/2, c]]
```

Sous rotation des axes tangents, ce tenseur devient `Rᵀ H R`. Sa norme de Frobenius reste constante : `||H||²_F = a²+b²/2+c²`. En revanche, la pénalité `a²+b²+c²` n'est pas cette norme. Pour conserver le même poids sur les parties constante et linéaire, nous testons donc `P=diag(1,1,1,1,1/2,1)` à la place de `I` dans `BᵀWB+λP`.

Cette modification ne prouve ni une précision supérieure, ni une amélioration du réseau. Elle fournit un contrôle d'invariance de l'opérateur MLS reconstruit en coordonnées 3D. La régularisation de l'interception est conservée : à `λ>0`, le gradient d'un signal constant n'est donc pas nécessairement exactement nul. Le test de reproduction des polynômes utilise `λ=0`.

Pour un stencil de gradient constitué de vecteurs `g_ij` à deux composantes, nous comparons aussi le normaliseur amont à

```text
s = max_i sum_j ||g_ij||₂
```

Cette borne ne change pas lorsque la base tangentielle est tournée. Elle borne la norme maximale des vecteurs gradients lorsque le signal scalaire est borné par 1. Les deux normaliseurs ne produisent pas nécessairement la même échelle, ce qui interdit d'en déduire une amélioration directe d'un réseau entraîné avec l'autre convention.

## Protocole enregistré avant exécution

Deux surfaces analytiques : sphère unité et selle `z=0.3(x²−y²)`. Le champ `f=sin(2x)+yz/4` possède un gradient tangent exact, obtenu par projection de son gradient ambiant. Sur la sphère, l'échantillonnage uniforme porte sur l'aire ; sur la selle il est uniforme dans son domaine `(x,y)`, pas dans l'aire de surface.

Pour chacune : 512 points, 20 voisins incluant le centre, trois graines `(0,1,2)`, distribution uniforme ou biaisée 90/10 entre deux moitiés, et bruit géométrique gaussien de paramètres `(0,0.005,0.02)` dans les unités de la surface. Le signal reste attaché aux points propres correspondants. Les normales exactes et estimées par PCA forment deux conditions distinctes. Les normales exactes en présence de bruit sont une ablation contrôlée, pas une information disponible sur un vrai scan.

Chaque condition utilise le même voisinage et le même signal avant et après une rotation indépendante de chaque base tangentielle. Cela représente **72 conditions répétées pour quatre méthodes**, soit 288 lignes. Les conditions ne sont pas 72 objets réels indépendants ; aucune significativité statistique inter-objets n'est revendiquée.

| Méthode | Pénalité | Normalisation | Largeur gaussienne |
| --- | --- | --- | --- |
| A | `λI`, convention amont | Convention amont | Globale |
| B | Tenseur quadratique | Convention amont | Globale |
| C | Tenseur quadratique | Borne invariante | Globale |
| D | Tenseur quadratique | Borne invariante | Moyenne locale |

A/B isole la régularisation ; B/C isole la normalisation ; C/D explore un traitement de la densité locale. La variante D reste une ablation exploratoire, pas une nouveauté revendiquée.

## Résultats CPU effectivement mesurés

Les fichiers [results.json](../results/cpu_reference/results.json) et [metrics.csv](../results/cpu_reference/metrics.csv) contiennent toutes les conditions, la configuration, les versions et les empreintes du code. Exécution NumPy en float64 : **288/288 conditions calculées, 8.67 secondes** sur l'environnement enregistré. Ce temps n'est pas une mesure GPU ni un benchmark comparatif de frameworks.

| Méthode | Erreur relative du gradient brut, moyenne | Dérive relative après rotation de base, gradient normalisé, moyenne | Dérive maximale |
| --- | ---: | ---: | ---: |
| A | 16.332 % | 2.081 % | 10.204 % |
| B | 16.392 % | 2.036 % | 10.333 % |
| C | 16.392 % | `1.09e-15` | `1.55e-15` |
| D | 17.385 % | `1.18e-15` | `2.02e-15` |

La pénalité tensorielle seule supprime la dérive du gradient **non normalisé** à la précision numérique, mais la normalisation amont maintient une dérive dans B. C corrige les deux effets dans ce test. La précision au gradient analytique reste comparable à A, légèrement inférieure en moyenne : ce résultat ne doit pas être présenté comme un gain de précision.

La largeur locale D améliore l'erreur agrégée sur les moitiés désignées « sparse » de 18.769 % à 17.141 %, tout en dégradant l'erreur globale. Cette moyenne inclut les distributions uniformes ; utiliser les lignes filtrées `density=biased` pour discuter exclusivement des zones effectivement sous-échantillonnées. Les résultats complets permettent cette distinction.

## Limites et reproduction officielle

Les valeurs mesurées concernent la transcription NumPy des formules du gradient. La comparaison numérique contre l'exécution du code PyTorch amont est prévue par `scripts/check_upstream.py` mais **n'a pas été exécutée ici**, car cet environnement ne possède pas PyTorch/PyG. Elle devra passer avant d'attribuer des valeurs numériques identiques au programme officiel. La démonstration sur la forme de la pénalité est indépendante de cette comparaison.

L'[environnement officiel](https://github.com/rubenwiersma/deltaconv/blob/186fec369fa2ceb4559830bc421282dddb2300a2/environment.yml) utilise Python 3.9 et PyTorch 1.11. Le notebook de reproduction le crée dans un environnement isolé, vérifie le commit et enregistre les packages effectivement résolus. Il s'agit d'une recette historique, pas d'un lock de tous les paquets transitifs. Son installation sur un Colab actuel reste à vérifier.

La [commande officielle ModelNet40](https://github.com/rubenwiersma/deltaconv/blob/186fec369fa2ceb4559830bc421282dddb2300a2/replication_scripts/pretrained/modelnet40.sh) évalue un checkpoint fourni dans le dépôt. Aucun résultat de classification n'est ajouté ici. Avant une étude de robustesse du réseau, figer le sous-ensemble de test, les nuages prétraités, leurs empreintes et les transformations. Une amélioration d'invariance numérique ne garantit pas un gain d'accuracy avec un checkpoint entraîné sous une autre normalisation.

Enfin, l'indépendance aux **bases tangentielles** ne doit pas être confondue avec l'invariance d'un classifieur aux rotations globales de ses entrées `xyz`. Le [modèle officiel](https://github.com/rubenwiersma/deltaconv/blob/186fec369fa2ceb4559830bc421282dddb2300a2/deltaconv/models/deltanet_base.py) utilise par défaut les positions comme caractéristiques scalaires ; ces caractéristiques changent lorsqu'on tourne l'objet. Le test de rotation rigide de notre gradient transporte aussi le champ scalaire et les normales, et ne constitue pas un test de classification.

## Contribution et crédits

DeltaConv, son architecture et ses opérateurs viennent de Wiersma, Nasikun, Eisemann et Hildebrandt. La contribution locale est un audit falsifiable, une petite modification explicite de deux conventions numériques, leurs ablations, des résultats CPU et un parcours de reproduction. Aucun entraînement, poids ou taux de succès officiel n'est présenté comme un résultat personnel. Voir [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

