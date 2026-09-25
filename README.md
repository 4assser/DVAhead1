# DVAhead1

Голова D.Va для веба и AR: `DvaGolovaOK_2_lite.glb` (glTF 2.0, основа — DAZ Genesis 8 Female, риг лица на 80 костей).
`DvaGolovaOK_2_lite.json` и `DvaGolovaOK_2_lite_base64.txt` — тот же GLB в base64.

- 📄 **[Анализ: скелет, готовность к ARKit, перенос в Google GNM](docs/ANALYSIS_RU.md)**
- 🛠 [Инструменты](tools/README.md): `python tools/inspect_glb.py DvaGolovaOK_2_lite.glb` — проверка скелета, морфов и 52 blendshape ARKit.

Итог анализа: риг лица хороший, но **blendshape ARKit нет (0/52)** — их нужно запечь из поз костей и починить материал глаз.
