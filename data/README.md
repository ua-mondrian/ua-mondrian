# Исходные данные

Файлы скопированы из UCI Machine Learning Repository без изменений. Контрольные суммы SHA-256 записаны в `code/results/manifest.json` при скачивании (11.11.2026) и продублированы в `SHA256SUMS.txt`. Проверка: `shasum -a 256 -c SHA256SUMS.txt`.

| Файл | Набор данных | Ссылка для цитирования | SHA-256 |
|---|---|---|---|
| `adult.data` | Adult | Becker B., Kohavi R. Adult. UCI Machine Learning Repository, 1996. https://doi.org/10.24432/C5XW20 | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` |
| `adult.test` | Adult | то же | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` |
| `bank-full.csv` | Bank Marketing | Moro S., Rita P., Cortez P. Bank Marketing. UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5K306 | `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` |
| `cmc.data` | Contraceptive Method Choice | Lim T.-S. Contraceptive Method Choice. UCI Machine Learning Repository, 1999. https://doi.org/10.24432/C59W2D | `ad2a49da55df24b061a994ecc4d90c856aba35acd05138851aa251c4f06303c9` |

Лицензия: наборы распространяются UCI Machine Learning Repository по лицензии Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. Правообладатели — авторы наборов, указанные выше. Авторы этого репозитория распространяют файлы без изменений с указанием источника.

Предобработка (удаление строк с «?» в Adult и т. п.) выполняется в коде (`code/data.py`); сами файлы в репозитории не меняются.
