# Article Classifier

Streamlit-приложение для multilabel-классификации научных статей по `title` и `abstract`.

Модель предсказывает категории arXiv (148 классов) и показывает:
- `Predicted labels` - метки выше порога `threshold`
- `Top-k labels` - самые вероятные метки

## Что в репозитории

- `app.py` - веб-интерфейс на Streamlit
- `model.py` - архитектура модели и конфиг
- `Классификатор_статей.ipynb` - ноутбук с экспериментами и обучением моделей
- `EXPERIMENTS_REPORT.md` - отчет по экспериментам и интерпретации результатов
- `best_arxiv_ultra_scibert/` - обученные артефакты (веса, tokenizer, config)
- `requirements.txt` - зависимости проекта

## Быстрый запуск

Требования: Python `3.12.13` 

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git lfs install
git lfs pull
streamlit run app.py --server.port 8505
```
