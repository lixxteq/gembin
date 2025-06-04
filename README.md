# Распознавание зависимостей исполняемого файла на основе анализа графа потока управления

[![ru](https://img.shields.io/badge/lang-ru-blue.svg)](https://github.com/lixxteq/gembin/blob/master/README.md)
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/lixxteq/gembin/blob/master/README_EN.md)

## Требования

- Linux / Cygwin on Windows
- Python 3.10-3.12
- radare2 > 5.8.x
- IDA Pro > 7.7 (при использовании драйвера IDAPython)

## Использование

### Извлечение атрибутов

```sh
python application/feature_extractor.py <executable path> -f <function to extract> -o <output ACFG file path>
```

### Предсказание схожести

```sh
python application/similarity_mp.py <lib ACFG file> <target ACFG file>
```

### Обучение модели

Требуется датасет в директории `data` и настройка формата наименований исполняемых файлов датасета в `config.py`.

```sh
python train.py --log_path <log path> --save_path <optional model path>
```

## Визуализация векторных представлений и предсказания схожести

> Опциональные зависимости `requirements.optional.txt` используются для демонстрации плоттинга 64-dim векторных представлений в интерфейсе gradio и предсказания их сходства c использованием pretrained model.

### Визуализация raw vectors

Генерация 64-dim векторов и запись в raw-формате:

```sh
python generate_emb.py --output_tsv embeddings.tsv demo_files/func1.json demo_files/func2.json ...
```

Визуализация в 3D пространстве, расчет косинусного сходства и расстояния Евклида:

```sh
python visualize_3d.py
```

> [!NOTE]
> Косинусное сходство и расстояние Евклида рассчитываются над raw-векторами и не используют inference, поэтому не показывают реальное сходство между функциями.

> [!NOTE]
> Для редуцирования в трехмерное пространство требуется как минимум 3 raw-вектора во входном файле.

### Визуализация similarity inference

Предсказание сходства между всеми функцями всех входных файлов с наборами ACFG:

```sh
python visualize_inference.py demo_files/func1.json demo_files/func2.json ...
```
