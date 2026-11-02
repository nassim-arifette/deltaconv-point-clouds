# Sources et crédits

L'architecture DeltaConv est due à Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann et Klaus Hildebrandt, *DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds*, ACM TOG / SIGGRAPH 2022, DOI 10.1145/3528223.3530166.

La baseline de cet audit transpose en NumPy les formules du gradient dans `deltaconv/geometry/grad_div_mls.py` du dépôt https://github.com/rubenwiersma/deltaconv au commit `186fec369fa2ceb4559830bc421282dddb2300a2`. Les méthodes B/C/D sont des modifications locales explicitement identifiées. La licence amont est conservée ci-dessous.

Les poids et jeux de données du notebook de reproduction restent ceux de leurs auteurs. Ils ne sont pas inclus dans ce dépôt.

## Licence amont

MIT License

Copyright (c) 2021 rubenwiersma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
