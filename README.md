## Medical assistance in the diagnosis of COVID-19
Based upon gathered data by the _Hospital Israelita Albert Einstein, São Paulo (Brazil),_ related to COVID-19 treated patients, available on [Kaggle](https://www.kaggle.com/dataset/e626783d4672f182e7870b1bbe75fae66bdfb232289da0a61f08c2ceb01cab01), there are several questions to be answered using machine learning, specifically Random Forest.

The available dataset is anonymous and it contains over than 5600 patient records with several exam results, patient admission, COVID-19 diagnosis.

The questions could be divided into two groups as follow:

1. Based on the labs data, is it possible to predict whether the patient condition is positive to COVID-19 or negative to COVID-19? Which is its accuracy? Which are the ten more important "questions"/variables to achieve this prediction?

2. Based on laboratory data, it is possible to predict whether the patient would be hospitalized at regular ward, at semi-intensive care unit, at intensive care unit or stay at home? Which are the more important "questions"/variables to achieve this prediction?

[See full project description](ref/proj3-iia-1-2020.pdf).

### Solution
Developed on [Python 3.8.5](https://www.python.org/downloads/release/python-385/) and you can run it as follow:

```
# Activate your virtual environment
python src/main.py
```
