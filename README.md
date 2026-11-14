# UA-Mondrian: учёт уязвимости атрибутов при выборе разреза в алгоритме Mondrian

Материалы к статье «Учёт уязвимости атрибутов при выборе разреза в алгоритме Mondrian: экспериментальная оценка компромисса полезности и риска для задач машинного обучения» (Р. А. Гайнутдинов, Е. С. Несмиянова; рукопись подана в журнал «Проблемы информационной безопасности. Компьютерные системы»).

В репозитории лежат протокол исследования, код, исходные данные, журналы всех прогонов и сводные таблицы. Из журналов воспроизводятся все таблицы и рисунки статьи.

*English summary — see [below](#english-summary).*

## Состав

| Путь | Содержимое |
|---|---|
| `protocol/` | Протокол исследования («План статьи.docx») и файл с его хэшем SHA-256, записанным 01.11.2026 ([подробнее](protocol/README.md)) |
| `data/` | Исходные файлы наборов UCI Adult, Bank Marketing и CMC в неизменном виде ([источники и лицензия](data/README.md)) |
| `code/*.py` | Реализация: UA-Mondrian и базовые методы, предобработка, атаки, метрики, статистика, запуск экспериментов |
| `code/tests/` | Модульные тесты протокола (20 тестов) |
| `code/logs/` | Журналы основного эксперимента: пилот, полный прогон, вторичные сценарии атак |
| `code/logs_posthoc/` | Журналы разведочных расчётов, не входивших в протокол |
| `code/results/` | Сводки по основному эксперименту и манифест прогона (версии ПО, SHA-256 данных, правила сидов) |
| `code/results_posthoc/` | Сводки разведочных расчётов |
| `code/analysis/` | Скрипты разведочных сводок и диагностики |
| `code/figures/` | Рисунки 1–5 статьи |

## Окружение

Расчёты выполнены на Python 3.14.2 на компьютере Apple M2 (8 ядер, 8 ГБ ОЗУ, macOS). Точные версии библиотек указаны в `requirements.txt` и `code/results/manifest.json`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

На macOS для xgboost нужен libomp (`brew install libomp`). `config.set_xgboost_runtime_env()` сам добавляет путь Homebrew в `DYLD_FALLBACK_LIBRARY_PATH`. Каталог с данными можно переопределить переменной окружения `UA_DATA_DIR`.

## Проверка без пересчёта

Все команды выполняются из `code/`.

```bash
(cd ../protocol && shasum -a 256 -c --ignore-missing SHA256SUMS.txt)   # хэш протокола
(cd ../data && shasum -a 256 -c SHA256SUMS.txt)                         # исходные данные
python -m pytest tests -q                                       # 20 тестов
python aggregate.py                                             # results/*.csv из logs/
python analysis/halfopen_summary.py                             # results_posthoc/*halfopen*.csv
python analysis/posthoc_summary.py                              # α-сетка, P_aux, R_rep
python analysis/risk_ci_posthoc.py                              # ДИ риска (исходная реализация атаки)
git status --short                                              # пусто = сводки совпали побайтно
```

Перед публикацией эти команды пересоздали 18 сводных файлов, и все совпали с опубликованными побайтно.

## Полный пересчёт

`run_full.py` дописывает только недостающие записи журналов. Поэтому для пересчёта с нуля сначала уберите опубликованные журналы, а потом сравните новые с ними.

| Шаг | Команда (из `code/`) | Время на Apple M2 |
|---|---|---|
| Пилот | `python run_pilot.py` | ≈ 6 мин |
| Полный прогон (повторы 0 и 1) | `python run_full.py` | ≈ 1,5 ч |
| Вторичные сценарии атак | `python run_full.py --secondary-only` | ≈ 1,5 ч |
| Сводки основного эксперимента | `python aggregate.py` | < 1 мин |
| Сетка α (разведочно) | `python run_posthoc_alpha.py` | ≈ 51 мин |
| Варианты P_aux (разведочно) | `python run_posthoc_paux.py` | ≈ 26 мин |
| Варианты R_rep (разведочно) | `python run_posthoc_rrep.py` | ≈ 29 мин |
| Атаки с полуоткрытыми интервалами | `python run_posthoc_attacks_halfopen.py` | не замерялось |
| Разведочные сводки | `python analysis/halfopen_summary.py`, `analysis/posthoc_summary.py`, `analysis/risk_ci_posthoc.py` | < 1 мин |
| Диагностика корня и Raw | `python analysis/root_scores.py`, `analysis/root_alpha_star.py`, `analysis/diag_raw_regularization.py` | минуты |
| Рисунки | `python draw_figures.py` | < 1 мин |

Пиковая память полного прогона не превышала 0,5 ГБ. Сплиты и все случайные величины заданы детерминированными сидами (`results/manifest.json`, `results/folds_manifest.csv`). Macro-F1 и метрики риска воспроизводятся точно. ROC-AUC и PR-AUC Random Forest при `n_jobs = -1` могут отличаться на 10⁻⁷–5·10⁻⁶ из-за порядка суммирования вероятностей.

## Где в статье какие файлы

| Элемент статьи | Файлы |
|---|---|
| Табл. 2 (проверка H1), рис. 2 | `results/h1_verdicts.csv`, `results/h1_pairs.csv`; столбец mean ΔLink — `results_posthoc/h1_dlink_halfopen.csv` |
| Табл. 3, рис. 3–4 (полезность, информационные потери) | `results/ml_summary.csv`, `results/privacy_summary.csv` |
| Табл. 4, рис. 5 (атаки, классовая оценка 1/∣EC∣) | `results_posthoc/attacks_halfopen_summary.csv`, `results_posthoc/risk_ci_halfopen.csv` |
| 3.1 (пилот, выполнимость, корень) | `results/pilot_summary.json`, `results/completeness.json`, `results/feasibility_overview.csv`, `results/root_screen_summary.csv` |
| 3.3 (B4 и Raw) | `results_posthoc/diag_raw_regularization*.csv` |
| 3.4 (вторичные сценарии атак) | `results/attacks_summary.csv` (исходная реализация), `results_posthoc/attacks_halfopen_summary.csv` |
| 3.5 (α, P_aux, R_rep, корень) | `results_posthoc/alpha_grid_summary_halfopen.csv`, `paux_summary.csv`, `rrep_summary_halfopen.csv`, `rrep_root_rho_summary.csv`; вывод `analysis/root_scores.py` и `analysis/root_alpha_star.py` |

## Отступления от протокола

Подтверждающий анализ (H1) выполнен строго по протоколу. Всё остальное — разведочные расчёты, и в статье они так и помечены. Отступления и исправления:

1. **Название компаратора B4.** В протоколе он назван «полным литературным InfoGain Mondrian». Реализован он по формуле протокола: атрибуты ранжируются по $IG_P(A_j) = H_P(Y) - H_P(Y \mid A_j)$, порог берётся от медианы. В исходном InfoGain Mondrian (LeFevre et al., 2008) выбирается разрез с минимальной взвешенной энтропией классов, поэтому в статье B4 описан как упрощение InfoGain Mondrian.
2. **Семантика интервалов в атаке.** В исходной реализации (`attacks.py`) границы числовых интервалов сравнивались как замкнутые, хотя правый потомок разреза открыт снизу. Атаки пересчитаны с полуоткрытыми интервалами (`attacks_halfopen.py`, `logs_posthoc/halfopen/`) на тех же деревьях, сплитах и сидах. Исходные журналы сохранены в `logs/runs_attacks.csv`. Вердикт H1 не меняется.
3. **Классовая оценка риска 1/∣EC∣** добавлена после эксперимента и вычисляется из имеющихся журналов.
4. **Разведочные расчёты** (сетка α, варианты P_aux и R_rep, регуляризация Raw, ДИ для риска) в протоколе не предусматривались и выполнены после основного прогона.
5. **Априорная вероятность в SAI-gain** вычисляется по всему набору (D_train ∪ D_test). Так это и описано в статье.

## Лицензии

- Код (`code/**/*.py`) распространяется по лицензии MIT, см. [`LICENSE`](LICENSE).
- Протокол, журналы, сводки и рисунки распространяются по лицензии CC BY 4.0, см. [`LICENSE-DATA.md`](LICENSE-DATA.md).
- Наборы данных UCI распространяются по лицензии CC BY 4.0 их правообладателей, см. [`data/README.md`](data/README.md).

## Как цитировать

См. [`CITATION.cff`](CITATION.cff); GitHub показывает кнопку «Cite this repository». Репозиторий: https://github.com/ua-mondrian/ua-mondrian. После выхода статьи сюда будут добавлены её выходные данные и DOI архивной копии.

## English summary

Code, data, pre-experiment protocol and complete run logs for the paper *Vulnerability-Aware Split Selection in Mondrian Anonymization: An Empirical Evaluation of the Utility–Risk Trade-off for Machine Learning* (manuscript submitted). The protocol document (`protocol/`, in Russian) was hashed with SHA-256 on 2026-11-01, before the pilot and the main runs on 2026-11-11/12. It was not deposited in a public registry before the experiments, so the public archive certifies that the document is unchanged, not when it was written. All result tables are regenerated byte-identically from `code/logs*` (`python aggregate.py` and `code/analysis/*.py`). The full re-run takes about 3 h on an Apple M2 laptop. Code: MIT; protocol, logs and results: CC BY 4.0; UCI datasets: CC BY 4.0 (original creators).
