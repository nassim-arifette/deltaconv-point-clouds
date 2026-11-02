"""Generate the CPU audit and the explicitly unverified official-replication notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


BOOTSTRAP = '''
import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/nassim-arifette/deltaconv-point-clouds.git"
here = Path.cwd()
project = next((p for p in (here, here.parent) if (p / "deltaconv_audit").is_dir()), None)
if project is None:
    if not IN_COLAB:
        raise RuntimeError("Run this notebook from the project or its notebooks directory.")
    project = Path("/content/deltaconv-point-clouds")
    if not project.exists():
        subprocess.run(["git", "clone", REPO_URL, str(project)], check=True)
    if not (project / "deltaconv_audit").is_dir():
        raise RuntimeError("The published project checkout is missing. Upload the project folder instead.")
if str(project) not in sys.path:
    sys.path.insert(0, str(project))
print("Project:", project)
'''


def save(cells, filename):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"name": filename, "provenance": []}
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, filename, "exec")
    out = ROOT / "notebooks" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(out)


def main():
    save([
        md('''# DeltaConv : audit CPU des bases tangentielles

Ce notebook vérifie les conventions numériques d'un gradient MLS, **pas un réseau entraîné**. Il lit les mesures
de référence et permet de relancer les 288 calculs sur deux surfaces analytiques. Aucun GPU ni téléchargement
de données n'est requis. Les quatre variantes sont détaillées dans `docs/research.md`.

Le clonage Colab demande que le dépôt cible soit publié. Localement, ouvrir ce notebook depuis le projet.'''),
        code(BOOTSTRAP),
        code('''
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==2.4.6"], check=True)
import json
reference = json.loads((project / "results/cpu_reference/results.json").read_text(encoding="utf-8"))
print("Type de preuve:", reference["evidence_kind"])
print("Conditions:", len(reference["rows"]), "| statut:", reference["status"])
for row in reference["summary"]:
    print(row["method"], "erreur gradient =", row["gradient_relative_l2_mean"],
          "dérive de base =", row["gauge_normalized_relative_l2_mean"])
'''),
        md('''## Relancer avec les paramètres enregistrés

Les normales exactes et PCA sont évaluées séparément. Chaque paire avant/après rotation utilise le même signal
et les mêmes voisins. Les données sont synthétiques : elles possèdent un gradient analytique de référence.
Les sorties sont placées dans un nouveau dossier pour conserver les mesures précédentes.'''),
        code('''
from datetime import datetime, timezone
POINTS = 512  #@param {type:"integer"}
output = project / "results/runs" / datetime.now(timezone.utc).strftime("notebook-%Y%m%d-%H%M%S")
subprocess.run([sys.executable, "-m", "deltaconv_audit.experiment", "--output", str(output),
                "--points", str(POINTS), "--seeds", "0", "1", "2"], cwd=project, check=True)
measured = json.loads((output / "results.json").read_text(encoding="utf-8"))
print("Mesures et provenance:", output)
'''),
        md('''## Examiner une condition sans confondre les moyennes

L'erreur au gradient analytique est mesurée avant normalisation. La dérive au changement de bases est mesurée
avant et après normalisation. Le filtre ci-dessous porte sur les régions réellement sous-échantillonnées.'''),
        code('''
rows = [r for r in measured["rows"] if r["density"] == "biased" and r["normal_source"] == "pca"
        and r["sigma"] == 0.02 and r["seed"] == 0]
for row in rows:
    print(row["shape"], row["method"], "erreur globale:", row.get("gradient_relative_l2"),
          "erreur zone rare:", row.get("sparse_gradient_relative_l2"),
          "dérive base:", row.get("gauge_normalized_relative_l2"))
'''),
        md('''## Interprétation

Une dérive faible indique que le choix des coordonnées ne modifie plus l'opérateur dans ces tests. Elle ne
prouve pas un gain de classification. La variante locale peut améliorer une région et dégrader l'ensemble ;
conserver ces deux résultats. Le second notebook compare la baseline NumPy au vrai code amont avant toute
étude du réseau, dans un environnement différent.'''),
    ], "01_operator_audit.ipynb")
    save([
        md('''# DeltaConv : parcours de reproduction officielle

**État : recette préparée, installation et inférence non exécutées dans l'environnement de travail actuel.**
Le code et les poids viennent des auteurs ; ce notebook ne réimplémente pas leur réseau.

Ce parcours requiert Linux x86_64 avec accès réseau ; Colab convient comme cible à vérifier. L'environnement
historique Python 3.9 / PyTorch 1.11 est créé séparément du noyau du notebook. Le solveur est fixé et son
empreinte vérifiée. Le paquet DeltaConv est ensuite réinstallé depuis le commit fixé, y compris son extension
native : CMake et un compilateur C++ doivent être disponibles. Les dépendances transitives résolues sont enregistrées, mais leur disponibilité future n'est
pas garantie. Le téléchargement ModelNet40 et son prétraitement prennent bien plus de temps que l'audit CPU.'''),
        code(BOOTSTRAP),
        code('''
work = project / "work/official"
subprocess.run([sys.executable, str(project / "scripts/setup_official.py"),
                "--workdir", str(work), "--install-env"], check=True)
official_python = work / "environment/bin/python"
official_repo = work / "DeltaConv"
'''),
        md('''## Vérifier la parité de la baseline

Ce contrôle appelle le gradient PyTorch officiel et la transcription NumPy avec les mêmes points, voisins,
bases et signaux, en float64. Les erreurs doivent rester sous `1e-8`. Un échec bloque l'attribution au code
officiel des mesures de la transcription. Aucun fichier de succès n'est fourni d'avance.'''),
        code('''
subprocess.run([str(official_python), str(project / "scripts/check_upstream.py"),
                "--upstream", str(official_repo)], check=True)
'''),
        md('''## Évaluer le checkpoint officiel sur ModelNet40

La commande suivante est celle du script officiel de reproduction, avec son checkpoint, et enregistre sa
sortie. Elle peut télécharger le jeu de données et traiter aussi les maillages d'entraînement lors de la
préparation. Les résultats qu'elle affichera seront les seuls résultats de classification de cette exécution.
En cas de panne du serveur de données historique, conserver l'erreur et installer les données officielles
manuellement ; ne pas substituer silencieusement un autre dataset.'''),
        code('''
import hashlib, json
checkpoint = official_repo / "experiments/pretrained_weights/modelnet40.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
command = [str(official_python), "experiments/train_modelnet.py", "--checkpoint", str(checkpoint),
           "--seed", "1", "--batch_size", "8", "--num_points", "1024", "--k", "20"]
log = work / "modelnet40-evaluation.log"
receipt = {"command": command, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           "status": "running"}
(work / "modelnet40-run.json").write_text(json.dumps(receipt, indent=2))
with log.open("w", encoding="utf-8") as handle:
    result = subprocess.run(command, cwd=official_repo, stdout=handle, stderr=subprocess.STDOUT)
receipt.update(status="ok" if result.returncode == 0 else "failed", returncode=result.returncode,
               log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(work / "modelnet40-run.json").write_text(json.dumps(receipt, indent=2))
print(log.read_text(encoding="utf-8")[-6000:])
result.check_returncode()
'''),
        md('''## Étape suivante, distincte des résultats actuels

Pour mesurer la robustesse du réseau : figer un sous-ensemble de test et des nuages prétraités avec leurs
empreintes, conserver le checkpoint, puis appliquer des perturbations appariées. Modifier les opérateurs
d'un checkpoint préentraîné change sa distribution d'activation ; comparer précision et invariance séparément.
Ne pas convertir les résultats analytiques de l'audit CPU en taux de réussite sur ModelNet40.'''),
    ], "02_official_replication.ipynb")


if __name__ == "__main__":
    main()
