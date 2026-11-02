# Site de recherche

Le site est une page unique en anglais, de style article interactif. Ses sources sont dans `web/` (Vite + TypeScript, Three.js, D3, KaTeX). La compilation produit `site/`, qui n'est pas versionné.

Les figures recalculent le gradient MLS dans le navigateur avec `web/src/mls.ts`, un portage TypeScript de `deltaconv_audit/operators.py`. `web/test/mls.test.ts` vérifie, sur les quatre nuages enregistrés et les quatre méthodes, que le portage reproduit les gradients NumPy (écart relatif < 1e-11), l'échelle de normalisation et les dérives mesurées.

## Mettre à jour les données

Depuis la racine du projet :

```bash
python tools/build_web_data.py   # entrées exactes des 4 nuages + 288 lignes -> web/public/data
python tools/build_site_data.py  # fichiers téléchargeables -> web/public/files
cd web
npm ci
npm test
npm run build                    # -> ../site
cd ..
python tools/check_site.py
```

`build_web_data.py` régénère les points, normales PCA, voisinages, signal et angles aléatoires avec les fonctions de l'audit, puis refuse l'export s'ils diffèrent de `results/cpu_reference/viewer_data.json`. `build_site_data.py` refuse tout audit incomplet avant de copier quoi que ce soit.

## Consulter localement

Le site charge ses données par `fetch` : il faut le servir en HTTP, l'ouverture directe de `index.html` ne suffit pas.

```bash
cd web && npm run dev                          # développement
python -m http.server 8000 --directory site    # version compilée
```

## Publier avec GitHub Pages

Le workflow [.github/workflows/pages.yml](../.github/workflows/pages.yml) installe les dépendances, lance les tests, compile `site/`, le vérifie puis le déploie. Dans le dépôt GitHub, choisir **Settings → Pages → Build and deployment → Source → GitHub Actions**.
