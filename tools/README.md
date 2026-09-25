# Инструменты анализа головы

## `inspect_glb.py` — быстрая проверка модели (нужен только numpy)

```bash
python tools/inspect_glb.py DvaGolovaOK_2_lite.glb        # также понимает .json-обёртку и _base64.txt
```

Что выводит:

- иерархию узлов с мировыми координатами костей;
- совпадение bind pose с rest pose;
- гистограмму влияний на вершину и кости без весов;
- меши, атрибуты и морф-таргеты;
- материалы (с предупреждением, если нет PBR-блока);
- размеры текстур;
- проверку наличия **всех 52 blendshape ARKit** с итогом «ГОТОВ / НЕ ГОТОВ».

Удобно гонять после экспорта из Blender.

## `experiments/` — рендеры, подгонка GNM, перенос мимики

Зависимости:

```bash
pip install numpy scipy pillow trimesh rtree h5py
git clone --depth 1 https://github.com/google/GNM ../GNM      # рядом с репозиторием, или export GNM_DIR=/путь/к/GNM
```

Запуск всего (~2 минуты на CPU):

```bash
bash tools/experiments/run_all.sh
```

| Скрипт | Что делает | Результат |
|---|---|---|
| `figures.py` | виды модели, схема скелета + сетка, позы костей (LBS), фрагмент текстуры лица | `docs/img/01…04_*.jpg` |
| `fit_gnm.py` | подгонка identity GNM Head v3 к D.Va: ориентиры → подобие → ICP + регуляризованный МНК, σ = 3 / 1 / 0.3 мм | `experiments/out/fit_sigma*.npz`, `fit_results.json` |
| `gnm_figures.py` | метрики подгонки, лист сравнения, перенос выражений GNM → сетка D.Va, измерение щели глаза | `docs/img/05_gnm_fit.jpg`, `06_gnm_expr_transfer.jpg`, `out/*.json`, `out/morph_*.npz` |
| `gnm_figures.py --naive` | тот же перенос без разделения губ (для сравнения) | `docs/img/06_gnm_expr_transfer_naive.jpg` |

Вспомогательные модули:

- `glb.py` — чтение GLB;
- `dva.py` — модель D.Va в numpy;
- `pose.py` — позы и LBS;
- `render.py` — простой z-buffer растеризатор на numpy, без GPU и Blender;
- `visibility.py` — видимость вершин;
- `gnm.py` — загрузчик GNM и numpy-порт semantic expression decoder (TensorFlow не нужен).

`experiments/out/morph_*.npz` — дельты вершин (`skin`, `teeth`, `eyes`) в системе координат D.Va. Их можно превратить в морф-таргеты. Это черновой перенос без корректоров век.
