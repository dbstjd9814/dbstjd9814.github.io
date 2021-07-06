# Assignment by YS
---

# Bike Sharing Demand

## 데이터분석과 시각화, 머신러닝 알고리즘으로 시간당 자전거 대여량을 예측하기

(이 쥬피터 노트북은 다음의 링크 https://bit.ly/ds-bike-0101 에서 다운받을 수 있습니다.)

이번 캐글 경진대회는 시간당 자전거 대여량을 예측하는 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) 입니다. 워싱턴 D.C 소재의 자전거 대여 스타트업 [Capital Bikeshare](https://www.capitalbikeshare.com/)의 데이터를 활용하여, 특정 시간대에 얼마나 많은 사람들이 자전거를 대여하는지 예측하는 것이 목표입니다.

사람들이 자전거를 대여하는데는 많은 요소가 관여되어 있을 겁니다. 가령 시간(새벽보다 낮에 많이 빌리겠죠), 날씨(비가 오면 자전거를 대여하지 않을 겁니다), 근무일(근무 시간에는 자전거를 대여하지 않겠죠) 등. 이런 모든 요소를 조합하여 워싱턴 D.C의 자전거 교통량을 예측해주세요. 이번 경진대회에서는 기존까지 배웠던 프로그래밍 언어와 인공지능&머신러닝 능력 외에도, 자전거 렌탈 시장에 대한 약간의 전문지식, 그리고 일반인의 기초 상식을 총동원 할 수 있습니다.

저번 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) 경진대회와 마찬가지로, 이번에도 프로그래밍 언어 파이썬([Python](https://www.python.org/)), 데이터 분석 패키지 판다스([Pandas](https://pandas.pydata.org/)), 그리고 머신러닝&인공지능 라이브러리인 싸이킷런([scikit-learn](scikit-learn.org))을 사용합니다. 여기에 더불어, 이번에는 데이터 시각화 패키지 [matplotlib](https://matplotlib.org/)와 [Seaborn](https://seaborn.pydata.org/)을 본격적으로 활용해볼 것입니다.

## 컬럼 설명

(데이터는 [다음의 링크](https://www.kaggle.com/c/bike-sharing-demand/data)에서 다운받으실 수 있습니다)

  * **datetime** - 시간. 연-월-일 시:분:초 로 표현합니다. (가령 2011-01-01 00:00:00은 2011년 1월 1일 0시 0분 0초)
  * **season** - 계절. 봄(1), 여름(2), 가을(3), 겨울(4) 순으로 표현합니다.
  * **holiday** - 공휴일. 1이면 공휴일이며, 0이면 공휴일이 아닙니다.
  * **workingday** - 근무일. 1이면 근무일이며, 0이면 근무일이 아닙니다.
  * **weather** - 날씨. 1 ~ 4 사이의 값을 가지며, 구체적으로는 다음과 같습니다.
    * 1: 아주 깨끗한 날씨입니다. 또는 아주 약간의 구름이 끼어있습니다.
    * 2: 약간의 안개와 구름이 끼어있는 날씨입니다.
    * 3: 약간의 눈, 비가 오거나 천둥이 칩니다.
    * 4: 아주 많은 비가 오거나 우박이 내립니다.
  * **temp** - 온도. 섭씨(Celsius)로 적혀있습니다.
  * **atemp** - 체감 온도. 마찬가지로 섭씨(Celsius)로 적혀있습니다.
  * **humidity** - 습도.
  * **windspeed** - 풍속.
  * **casual** - 비회원(non-registered)의 자전거 대여량.
  * **registered** - 회원(registered)의 자전거 대여량.
  * **count** - 총 자전거 대여랑. 비회원(casual) + 회원(registered)과 동일합니다.


```python
# 파이썬의 데이터 분석 패키지 Pandas(pandas.pydata.org) 를 읽어옵니다.
# Pandas는 쉽게 말해 파이썬으로 엑셀을 다룰 수 있는 툴이라고 보시면 됩니다.
# 이 패키지를 앞으로는 pd라는 축약어로 사용하겠습니다.
import pandas as pd
import numpy as np
```

## Load Dataset

언제나처럼 모든 데이터 분석의 시작은 주어진 데이터를 읽어오는 것입니다. [판다스(Pandas)](https://pandas.pydata.org/)의 [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)를 활용하여 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) 경진대회에서 제공하는 두 개의 데이터(train, test)를 읽어오겠습니다. ([다운로드 링크](https://www.kaggle.com/c/bike-sharing-demand/data))

앞서 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) 경진대회와 마찬가지로, 여기에서도 파일의 경로를 지정하는 방법에 주의하셔야 합니다. 만일 read_csv를 실행할 때 (**FileNotFoundError**)라는 이름의 에러가 난다면 경로가 제대로 지정이 되지 않은 것입니다. **파일의 경로를 지정하는 법이 생각나지 않는다면 [다음의 링크](http://88240.tistory.com/122)를 통해 경로를 지정하는 법을 복습한 뒤 다시 시도해주세요.**


```python
train = pd.read_csv("data/train.csv")

print(train.shape)
train.head()

# 데이터 개수가 커서, 통계적 유의성이 충분함
# datatime만 데이터 형태 변경하면 분석 가능
# NaN이 존재하지 않는다.
```

    (10886, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    10886 non-null  object 
     1   season      10886 non-null  int64  
     2   holiday     10886 non-null  int64  
     3   workingday  10886 non-null  int64  
     4   weather     10886 non-null  int64  
     5   temp        10886 non-null  float64
     6   atemp       10886 non-null  float64
     7   humidity    10886 non-null  int64  
     8   windspeed   10886 non-null  float64
     9   casual      10886 non-null  int64  
     10  registered  10886 non-null  int64  
     11  count       10886 non-null  int64  
    dtypes: float64(3), int64(8), object(1)
    memory usage: 1020.7+ KB



```python
test = pd.read_csv("data/test.csv")

print(test.shape)
test.head()
```

    (6493, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    6493 non-null   object 
     1   season      6493 non-null   int64  
     2   holiday     6493 non-null   int64  
     3   workingday  6493 non-null   int64  
     4   weather     6493 non-null   int64  
     5   temp        6493 non-null   float64
     6   atemp       6493 non-null   float64
     7   humidity    6493 non-null   int64  
     8   windspeed   6493 non-null   float64
    dtypes: float64(3), int64(5), object(1)
    memory usage: 456.7+ KB


## Preprocessing

데이터를 읽어왔으면, 이 데이터를 편하게 분석하고 머신러닝 알고리즘에 집어넣기 위해 간단한 전처리(Preprocessing) 작업을 진행하겠습니다.

[Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)는 편리하게도 대부분의 데이터가 전처리 되어있습니다. (가령 season 컬럼은 봄을 spring이라 표현하지 않고 1이라고 표현합니다) 그러므로 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) 경진대회와는 달리 간단한 전처리만 끝내면 바로 머신러닝 모델에 데이터를 집어넣을 수 있습니다.

### Parse datetime

먼저 **날짜(datetime)** 컬럼을 전처리 하겠습니다.

날짜 컬럼은 얼핏 보면 여러개의 숫자로 구성되어 있습니다. (ex: 2011-01-01 00:00:00) 하지만 결론적으로 숫자는 아니며, 판다스에서는 문자열(object) 또는 날짜(datetime64)로 인식합니다. (값에 하이픈(-)과 콜론(:)이 있기 때문입니다) 그러므로 날짜(datetime) 컬럼을 사용하기 위해서는 머신러닝 알고리즘이 이해할 수 있는 방식으로 전처리를 해줘야 합니다.

날짜(datetime) 컬럼을 전처리하는 가장 쉬운 방법은 연, 월, 일, 시, 분, 초를 따로 나누는 것입니다. 가령 2011-01-01 00:00:00은 2011년 1월 1일 0시 0분 0초라고 볼 수 있으므로, 2011, 1, 1, 0, 0, 0으로 따로 나누면 총 6개의 숫자가 됩니다. 즉, **날짜(datetime) 컬럼을 여섯개의 다른 컬럼으로 나누어주는 것이 날짜 컬럼을 전처리하는 핵심입니다**.


```python
train["datetime"] = pd.to_datetime(train["datetime"])

train["datetime-year"] = train["datetime"].dt.year
train["datetime-month"] = train["datetime"].dt.month
train["datetime-day"] = train["datetime"].dt.day
train["datetime-hour"] = train["datetime"].dt.hour
train["datetime-minute"] = train["datetime"].dt.minute
train["datetime-second"] = train["datetime"].dt.second

print(train.shape)
train[["datetime", "datetime-year", "datetime-month", "datetime-day", "datetime-hour", "datetime-minute", "datetime-second"]].head()
```

    (10886, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test["datetime"] = pd.to_datetime(test["datetime"])

test["datetime-year"] = test["datetime"].dt.year
test["datetime-month"] = test["datetime"].dt.month
test["datetime-day"] = test["datetime"].dt.day
test["datetime-hour"] = test["datetime"].dt.hour
test["datetime-minute"] = test["datetime"].dt.minute
test["datetime-second"] = test["datetime"].dt.second

print(test.shape)
test[["datetime", "datetime-year", "datetime-month", "datetime-day", "datetime-hour", "datetime-minute", "datetime-second"]].head()
```

    (6493, 15)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Explore

전처리(Preprocesing)를 끝냈으면 그 다음에는 데이터를 분석해보겠습니다.
주어진 데이터를 시각화나 분석 툴을 통해 다양한 관점에서 이해하는 과정을 탐험적 데이터 분석([Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis))이라고 합니다. 저번 타이타닉 문제와 마찬가지로, 이번에도 파이썬의 데이터 시각화 패키지인 ([matplotlib](https://matplotlib.org))와 [seaborn](https://seaborn.pydata.org/) 을 활용해서 분석해보겠습니다.


```python
# 불쾌지수 (Humidix)처럼 습도와 온도를 결합시켜 새로운 feature를 생성할 수 있음.# 불쾌지수 (Humidix)처럼 습도와 온도를 결합시켜 새로운 feature를 생성할 수 있음.

# matplotlib로 실행하는 모든 시각화를 자동으로 쥬피터 노트북에 띄웁니다.
# seaborn 도 결국에는 matplotlib를 기반으로 동작하기 때문에, seaborn으로 실행하는 모든 시각화도 마찬가지로 쥬피터 노트북에 자동적으로 띄워집니다.
%matplotlib inline

# 데이터 시각화 패키지 seaborn을 로딩합니다. 앞으로는 줄여서 sns라고 사용할 것입니다.
import seaborn as sns

# 데이터 시각화 패키지 matplotlib를 로딩합니다. 앞으로는 줄여서 plt라고 사용할 것입니다.
import matplotlib.pyplot as plt
```

### datetime

먼저 분석할 컬럼은 **날짜(datetime)** 컬럼입니다. 날짜 컬럼은 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) 경진대회의 핵심 컬럼이라고 볼 수 있으며, 이번 경진대회에서 상위 성적을 올리고 싶다면 날짜 컬럼을 완벽하게 이해하는 것이 무엇보다도 중요합니다.

먼저 연/월/일/시/분/초에 따른 자전거 대여량을 시각화 해보겠습니다.

### Q1-1) 시각화를 하기 전에 어떤 그림이 나올 것으로 예상하시나요? 최소 3가지 아이디어를 생각해보세요.

**주의**: 이 내용은 반드시 **시각화를 하기 전에 작성하셔야 합니다.** 그래야 시각화 결과와 본인의 아이디어를 비교해서 차이를 발견할 수 있습니다.

1. 일단 분(```Dates-minute```), 초(```Dates-second```)는 자전거 대여량을 판가름하는데 별 영향이 없을 것 같습니다. 가령 현재 시간이 37분이면 자전거를 대여하고, 43분이면 자전거를 대여하지 않는 행동을 하지는 않을 것입니다. 그러므로 countplot으로 시각화를 해보면, 마치 [Uniform Distribution](https://m.blog.naver.com/running_p/90179231685)과 같은 모양이 나올 것 같습니다.

1. 그리고 일(```Dates-day```)도 비슷합니다. 하지만 일(```Dates-day```)은 분과 초와는 다르게, 1) 2월에는 28일 이후가 존재하지 않기 때문에, 29, 30, 31일은 다른 날보다 데이터가 적을 수도 있습니다. (예외적으로 2012년은 2월 29일이 있습니다), 비슷하게 2) 31일의 경우에는 다른 날에 비해 데이터가 절반밖에 되지 않을 것입니다. 하지만 우리는 데이터의 갯수보다는 날짜별 자전거의 평균 대여량이 중요하기 때문에, 실제 분석에는 큰 영향을 미치지 않을 것으로 예상합니다.

1. 이런 사항 외에도, 사람의 행동 패턴 상으로 날짜나 시간이라는 개념이 자전거를 대여하는데 중요한 영향을 미칠 것 같습니다. 가령 1) 시간(hour)을 기준으로 새벽보다는 오후에 사람들이 자전거를 많이 빌릴것이며, 2) 월(month)을 기준으로 추운 겨울보다는 따뜻한 봄이나 가을, 내지는 더운 여름이 더 많이 빌릴 것 같습니다.



자, 그럼 위 예상과 실제 데이터가 일치하는지 데이터 시각화를 통해 살펴보도록 하겠습니다.


```python
# 여러개 한번에 시각화 하기
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows = 2, ncols = 3)
figure.set_size_inches(18, 8)

sns.barplot(data=train, x="datetime-year", y="count", ax=ax1)
sns.barplot(data=train, x="datetime-month", y="count", ax=ax2)
sns.barplot(data=train, x="datetime-day", y="count", ax=ax3)
sns.barplot(data=train, x="datetime-hour", y="count", ax=ax4)
sns.barplot(data=train, x="datetime-minute", y="count", ax=ax5)
sns.barplot(data=train, x="datetime-second", y="count", ax=ax6)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7ae994640>




![png](output_17_1.png)


### Q1-2) 이 시각화로 발견할 수 있는 사실은 어떤 게 있을까요? 그리고 앞서 우리의 예상과 어떤 차이가 있나요?

**datetime-year**
  * 2011년도의 자전거 대여량보다 2012년도의 자전거 대여량이 더 높습니다. 이는 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) 경진대회를 주최한 [Capital Bikeshare](https://www.capitalbikeshare.com/)사가 꾸준히 성장하고 있다고 간주할 수 있습니다.

**datetime-month**
  * 주로 여름(6-8월)에 자전거를 많이 빌리며, 겨울(12-2월)에는 자전거를 많이 빌리지 않습니다.
  * 같은 겨울이라도 12월의 자전거 대여량이 1월의 자전거 대여량보다 두 배 가까이 높아 보입니다. 하지만 여기에는 숨겨진 비밀이 있는데, 다음에 나올 다른 시각화에서 자세히 살펴보겠습니다.
  
**datetime-day**
  * x축을 자세히 보면 1일부터 19일까지밖에 없습니다. 20일은 어디에 있을까요? 바로 test 데이터에 있습니다. 이 시각화에서 알 수 있는 내용은, train 데이터와 test 데이터를 나누는 기준이 되는 컬럼이 바로 ```datetime-day```라는 것입니다. 그러므로 21일 이후의 자전거 대여량에 대해서는 우리도 알 수 없고, 머신러닝 알고리즘도 알지 못할 것입니다.

**datetime-hour**
  * 새벽 시간에는 사람들이 자전거를 빌리지 않으며, 오후 시간에 상대적으로 자전거를 많이 빌립니다.
  * 특이하게도 두 부분에서 사람들이 자전거를 특별히 많이 빌리는 현상이 있습니다. 바로 출근 시간(7-9시)과 퇴근 시간(16시-19시) 입니다.
  * 물론 출퇴근시간이 아닌 다른 시간대에 자전거를 빌리는 경우도 존재합니다. 이는 다음에 나올 다른 시각화에서 자세히 살펴보겠습니다.

**datetime-minute** & **datetime-second**
  * 이 두 컬럼은 x축이 모두 0으로 되어있습니다. 즉, **datetime-minute**과 **datetime-second**은 기록되고 있지 않다는 사실을 알 수 있습니다.

자, 이제 더 중요한 사실에 대해서 고민해 보도록 하겠습니다.

우리에게 중요한건 데이터에 어떤 특징이 있는지 발견하는 것도 있지만, **이 특징을 활용해 앞으로 사용할 머신러닝 알고리즘을 개선시킬 수 있는가?**가 더 중요합니다. 또한 개선을 한다면 구체적으로 어떤 방식으로 개선하는지도 중요하겠죠.

### Q1-3) 이 사실을 통해 어떻게 예측 모델을 개선할 수 있을까요? 최소 3가지 아이디어를 내보세요.

1. 먼저 분(```datetime-minute```)과 초(```datetime-second```)는 기록되지 않기 때문에 굳이 사용할 필요가 없을 것 같습니다. 차후에 머신러닝 알고리즘에 적용할 때, 이 부분은 feature에서 제거해도 될 것 같습니다.
2. 앞서 설명한대로, train 데이터와 test 데이터를 나누는 기준이 되는 컬럼이 바로 일(```datetime-day```) 컬럼입니다. 이런 경우 **datetime-day**를 feature로 집어넣으면 머신러닝 알고리즘이 과적합([overfitting](https://hyperdot.wordpress.com/2017/02/06/%EA%B3%BC%EC%A0%81%ED%95%A9overfitting/)) 되는 현상이 일어날 수 있습니다. 그러므로 train 데이터와 test 데이터를 나누는 기준이 되는 컬럼이 있으면, 이 컬럼은 feature로 사용하지 않는 것이 좋을 것 같습니다.
3. 이외에도 시(```datetime-hour```)컬럼을 보면 출퇴근시간에 사람들이 자전거를 많이 빌린다는 사실을 알 수 있습니다. 그렇다면, 만일 머신러닝 알고리즘이 출퇴근시간이라는 개념을 이해하지 못한다고 하면 이를 별도의 feature로 넣어주면 성능 향상을 꾀할 수 있을 듯 합니다. (다만 아쉽게도, ```workingday```라는 컬럼이 이 역할을 대신하고 있을 것입니다)
---

### weather 컬럼 분석

그 다음 분석하고 싶은 컬럼은 날씨를 나타내는 ```weather``` 컬럼입니다. 이 컬럼을 다음의 값을 가지며, 구체적인 설명은 다음과 같습니다.

  * 1: 아주 깨끗한 날씨입니다. 또는 아주 약간의 구름이 끼어있습니다.
  * 2: 약간의 안개와 구름이 끼어있는 날씨입니다.
  * 3: 약간의 눈, 비가 오거나 천둥이 칩니다.
  * 4: 아주 많은 비가 오거나 우박이 내립니다.

이 데이터를 엑셀 분석, 내지는 시각화하여 weather에 따라 자전거 대여량이 어떻게 변하는지 살펴보도록 하겠습니다.



### Q2-1) 시각화를 하기 전에 어떤 그림이 나올 것으로 예상하시나요? 최소 3가지 아이디어를 생각해보세요.

**주의**: 이 내용은 반드시 **시각화를 하기 전에 작성하셔야 합니다.** 그래야 시각화 결과와 본인의 아이디어를 비교해서 차이를 발견할 수 있습니다.

1. 일단 당연하지만 안 좋은 날씨일수록 자전거 대여량이 낮아질 것 같습니다. 1(깨끗한 날씨)의 경우보다 4(아주 많은 비나 우박이 오는 날씨)인 경우에 자전거를 덜 빌릴 것입니다.
2. 그리고 값이 숫자(1, 2, 3, 4)로 되어있지만, 실제로는 수의 높고 낮은 관계가 존재하지 않을 것입니다. (이를 전문용어로 연속형(continuous) 데이터 vs 범주형(categorical) 데이터라고 합니다) 그러므로 보이는 것과는 다르게, 실제로는 범주형(categorical) 데이터로 처리해야 할 것입니다.
3. 아주 심하진 않겠지만, 날씨마다의 편차가 있을 것입니다. 가령 어떤 날은 날씨가 좋아도 안 빌리고, 어떤 날은 날씨가 안 좋아도 많이 빌릴 수도 있습니다.

이번에도 위 예상과 실제 데이터가 일치하는지 데이터 시각화를 통해 살펴보도록 하겠습니다.


```python
sns.barplot(data=train, x="weather", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7ae830bb0>




![png](output_24_1.png)



```python
train[train["weather"]==4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5631</th>
      <td>2012-01-09 18:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>8.2</td>
      <td>11.365</td>
      <td>86</td>
      <td>6.0032</td>
      <td>6</td>
      <td>158</td>
      <td>164</td>
      <td>2012</td>
      <td>1</td>
      <td>9</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test[test["weather"]==4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>2011-01-26 16:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>9.02</td>
      <td>9.85</td>
      <td>93</td>
      <td>22.0028</td>
      <td>2011</td>
      <td>1</td>
      <td>26</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3248</th>
      <td>2012-01-21 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>5.74</td>
      <td>6.82</td>
      <td>86</td>
      <td>12.9980</td>
      <td>2012</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train["count"][train["weather"]==3].describe()
```




    count    859.000000
    mean     118.846333
    std      138.581297
    min        1.000000
    25%       23.000000
    50%       71.000000
    75%      161.000000
    max      891.000000
    Name: count, dtype: float64



### Q2-2) 이 시각화로 발견할 수 있는 사실은 어떤 게 있을까요? 그리고 앞서 우리의 예상과 어떤 차이가 있나요?

1. 앞서 생각한대로 날씨(```weather```)가 안 좋을수록 자전거 대여량이 낮아지는 현상을 발견할 수 있었습니다. 즉, 날씨(```weather```)값이 3보다 2가, 2보다 1이 더 자전거를 많이 빌리는 현상이 보입니다.
2. 하지만 굉장히 특이하게도 날씨가 4인 경우, 즉 아주 많은 비가 오거나 우박이 내리는 경우에 자전거를 많이 빌리는 현상이 보입니다. 심지어는 날씨가 2인 경우(약간의 안개나 구름)에 못지 않게 자전거를 많이 빌리는 사실을 알 수 있습니다.
3. 그리고 시각화에서 신뢰 구간(confidence interval)을 상징하는 검은색 세로 선이 날씨가 4인 경우에는 보이지 않습니다. 추측컨데 날씨가 4인 경우에는 일반적인 분포와는 다소 다른 현상이 일어나고 있다고 판단할 수 있습니다.

### Q2-3) 이 사실을 통해 어떻게 예측 모델을 개선할 수 있을까요? 최소 3가지 아이디어를 내보세요.

1. 날씨(```weather```) 컬럼값이 1 ~ 3인 것만 봤을 때, 이 컬럼을 머신러닝 알고리즘에 feature로 넣으면 우리가 별도의 룰을 설정해주지 않아도 머신러닝 알고리즘이 알아서 날씨(```weather```)에 따른 자전거 대여량의 변화량을 예측할 수 있을 것 같습니다. 아마도 날씨가 좋을 수록(1에 가까울수록) 자전거를 많이 빌리고, 안 좋을수록(3에 가까울수록) 자전거를 덜 빌릴 것 같습니다.
2. 날씨(```weather```) 컬럼값이 4인 곳에서는 1,2,3의 흐름과는 다르게 독특한 값이 발생했고, 신뢰구간 검은 세로선이 없는 것으로 보아 데이터가 한 개 혹은 중복값을 가질 것으로 보고, 어떤 데이터인지 살펴보았다. 날씨(```weather```)가 4인 데이터는 단 한 개가 존재했으며, 2012-01-09 18:00:00 이며, 휴일이 아닌 일하는 날의 퇴근시간인 것으로 파악되었다. 10886개의 데이터 중 한 개이기 때문에 이 데이터는 **무시하는 것**이 좋을 것 같다. 날씨 컬럼이 3인 곳에서 최대 count 값은 891로 날씨 4인 데이터의 count 값인 164보다 훨씬 크므로, 날씨 4는 날씨 3으로 묶기로 한다.
3. 날씨(```weather```) 컬럼은 범주형 데이터로 활용할 것이므로, One Hot Coding을 사용할 것이다. 또한 직관적으로 보기위해 날씨 1은 ```weather_Clear```컬럼에 True , 날씨 2는 ```weather_Cloudy```컬럼에 True, 날씨 3,4는 ```weather_Bad```컬럼에 True로 하기로 한다. 


```python
train["weather_Clear"] = train["weather"]==1
train["weather_Cloudy"] = train["weather"]==2
train["weather_Bad"] = (train["weather"]==3) | (train["weather"]==4)

print(train.shape)
train.loc[train["weather"]==4, ["weather", "weather_Clear", "weather_Cloudy", "weather_Bad"]]
```

    (10886, 21)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weather</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5631</th>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
test["weather_Clear"] = test["weather"]==1
test["weather_Cloudy"] = test["weather"]==2
test["weather_Bad"] = (test["weather"]==3) | (test["weather"]==4)

print(test.shape)
test.loc[test["weather"]==4, ["weather", "weather_Clear", "weather_Cloudy", "weather_Bad"]]
```

    (6493, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weather</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3248</th>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



#### weather_Clear, weather_Cloudy, weather_Bad 세 가지 컬럼을 사용하도록 한다.
---

### windspeed 컬럼 분석

그 다음 분석하고 싶은 컬럼은 날씨를 나타내는 풍속을 나타내는 ```windspeed``` 컬럼입니다. 이 컬럼은 0에서 56까지의 값을 가집니다. 이 데이터도 시각화 해보도록 하겠습니다.

### Q3-1) 시각화를 하기 전에 어떤 그림이 나올 것으로 예상하시나요? 최소 3가지 아이디어를 생각해보세요.

1. 이 데이터는 연속형(continuous) 자료이므로 분포를 시각화하면 전형적인 [정규 분포](https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C_%EB%B6%84%ED%8F%AC)가 나올 것입니다.
2. 하지만 이 데이터는 현실 세계의 데이터이기 때문에, 이론처럼 완벽한 정규 분포가 나오지는 않을 것입니다. 아마도 추측컨데 1) 몇몇 아웃라이어가 존재하거나, 2) 바람이 특별하게 많이 불어서 분포의 오른쪽이 길게 늘어지는 현상이 생길 것 같습니다.
3. 그리고 추측컨데 바람이 너무 많이 불면 사람들이 자전거를 덜 빌릴 것으로 예상합니다.

위 예상과 실제 데이터가 일치하는지 다시 한 번 살펴보도록 하겠습니다.


```python
plt.figure(figsize = (18, 4))

sns.distplot(train["windspeed"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7aed70520>




![png](output_36_1.png)



```python
plt.figure(figsize = (18, 4))

sns.pointplot(data=train, x="windspeed", y="count")

# 모수가 부족한 구간 = 삐죽삐죽하고, 신뢰구간 세로선이 김.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7af729910>




![png](output_37_1.png)



```python
pd.pivot_table(train, index="windspeed", values="count", aggfunc=["mean","count","min","max"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>windspeed</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0000</th>
      <td>161.101295</td>
      <td>1313</td>
      <td>1</td>
      <td>970</td>
    </tr>
    <tr>
      <th>6.0032</th>
      <td>147.864679</td>
      <td>872</td>
      <td>1</td>
      <td>782</td>
    </tr>
    <tr>
      <th>7.0015</th>
      <td>169.852031</td>
      <td>1034</td>
      <td>1</td>
      <td>888</td>
    </tr>
    <tr>
      <th>8.9981</th>
      <td>175.645536</td>
      <td>1120</td>
      <td>1</td>
      <td>872</td>
    </tr>
    <tr>
      <th>11.0014</th>
      <td>202.262062</td>
      <td>1057</td>
      <td>1</td>
      <td>900</td>
    </tr>
    <tr>
      <th>12.9980</th>
      <td>202.249520</td>
      <td>1042</td>
      <td>1</td>
      <td>943</td>
    </tr>
    <tr>
      <th>15.0013</th>
      <td>210.833507</td>
      <td>961</td>
      <td>1</td>
      <td>948</td>
    </tr>
    <tr>
      <th>16.9979</th>
      <td>214.847087</td>
      <td>824</td>
      <td>1</td>
      <td>977</td>
    </tr>
    <tr>
      <th>19.0012</th>
      <td>218.051775</td>
      <td>676</td>
      <td>1</td>
      <td>892</td>
    </tr>
    <tr>
      <th>19.9995</th>
      <td>225.235772</td>
      <td>492</td>
      <td>1</td>
      <td>968</td>
    </tr>
    <tr>
      <th>22.0028</th>
      <td>185.053763</td>
      <td>372</td>
      <td>1</td>
      <td>890</td>
    </tr>
    <tr>
      <th>23.9994</th>
      <td>220.010949</td>
      <td>274</td>
      <td>1</td>
      <td>856</td>
    </tr>
    <tr>
      <th>26.0027</th>
      <td>228.744681</td>
      <td>235</td>
      <td>1</td>
      <td>837</td>
    </tr>
    <tr>
      <th>27.9993</th>
      <td>219.363636</td>
      <td>187</td>
      <td>2</td>
      <td>704</td>
    </tr>
    <tr>
      <th>30.0026</th>
      <td>217.171171</td>
      <td>111</td>
      <td>1</td>
      <td>834</td>
    </tr>
    <tr>
      <th>31.0009</th>
      <td>208.955056</td>
      <td>89</td>
      <td>1</td>
      <td>643</td>
    </tr>
    <tr>
      <th>32.9975</th>
      <td>184.075000</td>
      <td>80</td>
      <td>1</td>
      <td>857</td>
    </tr>
    <tr>
      <th>35.0008</th>
      <td>230.155172</td>
      <td>58</td>
      <td>1</td>
      <td>770</td>
    </tr>
    <tr>
      <th>36.9974</th>
      <td>197.045455</td>
      <td>22</td>
      <td>21</td>
      <td>672</td>
    </tr>
    <tr>
      <th>39.0007</th>
      <td>176.888889</td>
      <td>27</td>
      <td>1</td>
      <td>755</td>
    </tr>
    <tr>
      <th>40.9973</th>
      <td>189.363636</td>
      <td>11</td>
      <td>7</td>
      <td>366</td>
    </tr>
    <tr>
      <th>43.0006</th>
      <td>137.916667</td>
      <td>12</td>
      <td>5</td>
      <td>342</td>
    </tr>
    <tr>
      <th>43.9989</th>
      <td>192.375000</td>
      <td>8</td>
      <td>45</td>
      <td>597</td>
    </tr>
    <tr>
      <th>46.0022</th>
      <td>67.333333</td>
      <td>3</td>
      <td>3</td>
      <td>185</td>
    </tr>
    <tr>
      <th>47.9988</th>
      <td>140.500000</td>
      <td>2</td>
      <td>1</td>
      <td>280</td>
    </tr>
    <tr>
      <th>50.0021</th>
      <td>171.000000</td>
      <td>1</td>
      <td>171</td>
      <td>171</td>
    </tr>
    <tr>
      <th>51.9987</th>
      <td>5.000000</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>56.9969</th>
      <td>269.500000</td>
      <td>2</td>
      <td>181</td>
      <td>358</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[train["windspeed"] > 43.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>...</th>
      <th>count</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>750</th>
      <td>2011-02-14 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22.96</td>
      <td>26.515</td>
      <td>21</td>
      <td>43.9989</td>
      <td>19</td>
      <td>...</td>
      <td>90</td>
      <td>2011</td>
      <td>2</td>
      <td>14</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>757</th>
      <td>2011-02-14 22:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.94</td>
      <td>14.395</td>
      <td>46</td>
      <td>43.9989</td>
      <td>1</td>
      <td>...</td>
      <td>45</td>
      <td>2011</td>
      <td>2</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>760</th>
      <td>2011-02-15 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>12.30</td>
      <td>12.120</td>
      <td>42</td>
      <td>51.9987</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>2011</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>761</th>
      <td>2011-02-15 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11.48</td>
      <td>11.365</td>
      <td>41</td>
      <td>46.0022</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>2011</td>
      <td>2</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>862</th>
      <td>2011-02-19 09:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>16.40</td>
      <td>20.455</td>
      <td>16</td>
      <td>43.9989</td>
      <td>18</td>
      <td>...</td>
      <td>55</td>
      <td>2011</td>
      <td>2</td>
      <td>19</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>867</th>
      <td>2011-02-19 14:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>18.86</td>
      <td>22.725</td>
      <td>15</td>
      <td>43.9989</td>
      <td>102</td>
      <td>...</td>
      <td>196</td>
      <td>2011</td>
      <td>2</td>
      <td>19</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>868</th>
      <td>2011-02-19 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>18.04</td>
      <td>21.970</td>
      <td>16</td>
      <td>50.0021</td>
      <td>84</td>
      <td>...</td>
      <td>171</td>
      <td>2011</td>
      <td>2</td>
      <td>19</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2755</th>
      <td>2011-07-03 17:00:00</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>32.80</td>
      <td>37.120</td>
      <td>49</td>
      <td>56.9969</td>
      <td>181</td>
      <td>...</td>
      <td>358</td>
      <td>2011</td>
      <td>7</td>
      <td>3</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2756</th>
      <td>2011-07-03 18:00:00</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>32.80</td>
      <td>37.120</td>
      <td>49</td>
      <td>56.9969</td>
      <td>74</td>
      <td>...</td>
      <td>181</td>
      <td>2011</td>
      <td>7</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5482</th>
      <td>2012-01-03 13:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>7.38</td>
      <td>6.060</td>
      <td>34</td>
      <td>43.9989</td>
      <td>5</td>
      <td>...</td>
      <td>73</td>
      <td>2012</td>
      <td>1</td>
      <td>3</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6132</th>
      <td>2012-02-11 18:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>9.02</td>
      <td>9.090</td>
      <td>47</td>
      <td>43.9989</td>
      <td>3</td>
      <td>...</td>
      <td>108</td>
      <td>2012</td>
      <td>2</td>
      <td>11</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6141</th>
      <td>2012-02-12 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4.10</td>
      <td>2.275</td>
      <td>46</td>
      <td>46.0022</td>
      <td>0</td>
      <td>...</td>
      <td>14</td>
      <td>2012</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6142</th>
      <td>2012-02-12 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4.10</td>
      <td>2.275</td>
      <td>46</td>
      <td>47.9988</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2012</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6444</th>
      <td>2012-03-05 18:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>11.48</td>
      <td>11.365</td>
      <td>55</td>
      <td>43.9989</td>
      <td>12</td>
      <td>...</td>
      <td>375</td>
      <td>2012</td>
      <td>3</td>
      <td>5</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6513</th>
      <td>2012-03-08 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>26.24</td>
      <td>31.060</td>
      <td>38</td>
      <td>46.0022</td>
      <td>24</td>
      <td>...</td>
      <td>185</td>
      <td>2012</td>
      <td>3</td>
      <td>8</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6515</th>
      <td>2012-03-08 17:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>25.42</td>
      <td>31.060</td>
      <td>38</td>
      <td>43.9989</td>
      <td>52</td>
      <td>...</td>
      <td>597</td>
      <td>2012</td>
      <td>3</td>
      <td>8</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6988</th>
      <td>2012-04-09 12:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22.14</td>
      <td>25.760</td>
      <td>28</td>
      <td>47.9988</td>
      <td>94</td>
      <td>...</td>
      <td>280</td>
      <td>2012</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 21 columns</p>
</div>




```python
plt.figure(figsize = (18, 4))

sns.pointplot(data=train[train["windspeed"] < 43.5], x="windspeed", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7b041eb50>




![png](output_40_1.png)


### Q3-2) 이 시각화로 발견할 수 있는 사실은 어떤 게 있을까요? 그리고 앞서 우리의 예상과 어떤 차이가 있나요?

1. 분포는 전반적으로 정규 분포가 나오지만, 1) 값이 연속적(continuous)이지 않고 듬성듬성 떨어져 있습니다. 즉, 이 데이터는 연속형(continuous) 데이터가 아닌 범주형(categorical) 데이터에 가까워 보입니다.
2. 더 특이한건, 풍속이 0인 경우가 굉장히 많으며, 정규 분포가 이상하게 보일 정도로 비중이 높습니다.
3. 또한 풍속이 과하게 높을수록 자전거를 덜 빌리는 현상이 보이는 것 같은데, 이는 전반적으로 모수가 부족한 듯 하여 신뢰도가 높지 않습니다. 다만 풍속이 낮을 경우에 전반적으로 자전거 대여량이 낮은 현상이 보입니다. (이는 우리가 예상하지 못한 현상입니다)

### Q3-3) 이 사실을 통해 어떻게 예측 모델을 개선할 수 있을까요? 최소 3가지 아이디어를 내보세요.

1. 이 풍속(windspeed) 데이터를 머신러닝 알고리즘에 집어넣으면 머신러닝 알고리즘의 풍속에 따른 자전거 대여량의 변화를 스스로 판단할 수 있을 것 같습니다. 더 정확히는, 풍속이 낮거나 높을수록 자전거를 덜 빌리고, 풍속이 적당할 때 자전거를 더 많이 빌린다는 사실을 알 수 있습니다.
2. 풍속이 0인 경우가 1313건이나 되는걸로 보아 풍속측정기기가 잘못 인식하는 것으로 보인다. 결측값을 0으로 계산했을 수도 있고, 특정 풍속 이하는 풍속계량기가 받아들이지 못해 0으로 일관되게 측정되었을 수도 있다. 다른 컬럼들과의 관계를 이용해서 풍속이 0인 곳을 채우기로 한다. 풍속이 실제로 0으로 관측될 확률은 매우매우 적다.
3. 모수의 수가 너무 적어서 신뢰가 떨어지는 구간은 삭제하기로 한다. (windspeed > 43.5 구간 삭제)


```python
# 2에서 언급한 풍속 0인 곳 채우기
from sklearn.ensemble import RandomForestRegressor

dataWind0 = train[train["windspeed"]==0]
dataWindNot0 = train[train["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","datetime-month","temp","datetime-year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
train = dataWindNot0.append(dataWind0)
train.reset_index(inplace=True)
train.drop('index',inplace=True,axis=1)
```

    <ipython-input-11-ba29b41ec222>:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      dataWind0["windspeed"] = wind0Values



```python
print(train.shape)
train.loc[train["windspeed"] == 0]
```

    (10886, 21)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>...</th>
      <th>count</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 21 columns</p>
</div>




```python
# 3에서 언급한 신뢰부족한 데이터 삭제
train = train.loc[train["windspeed"] < 43.5]

print(train.shape)
train.loc[train["windspeed"] > 43.5]
```

    (10869, 21)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>...</th>
      <th>count</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 21 columns</p>
</div>



### temp

이번에 분석할 데이터는 온도(```temp```) 컬럼입니다. 여기서부터는 제가 직접 하지 않고, 여러분들을 위한 과제로 제공하겠습니다. 앞서 컬럼들을 분석했던 것 처럼, 온도(```temp```) 컬럼도 직접 분석해보세요. 

힌트: 온도(```temp```) 컬럼만으로 좋은 분석 결과가 나오지 않는다면, 체감온도(```atemp```)를 포함한 다른 컬럼을 활용하여 시각화해보세요. 시각화는 [lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html?highlight=lmplot#seaborn.lmplot)이나 [scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html?highlight=scatterplot#seaborn.scatterplot)을 사용하면 직관적인 시각화를 할 수 있을 것입니다. (단 ```scatterplot```은 seaborn의 버전이 낮으면 실행되지 않으니 이 점 주의해주세요. 이 경우는 버전을 업그레이드 한 뒤 사용하시면 됩니다)

### Q4-1) 온도(```temp```) 컬럼을 시각화 하기 전에 어떤 그림이 나올 것으로 예상하시나요?
주의: 이 내용은 반드시 시각화를 하기 전에 작성하셔야 합니다. 그래야 시각화 결과와 본인의 아이디어를 비교해서 차이를 발견할 수 있습니다.

1. 온도가 너무 높거나 너무 낮으면, 추측컨데, 자전거 이용률이 낮을 가능성이 높을 것이다. 하지만 온도가 보통이더라도 여름에 비가오거나 겨울에 화창할때 온도가 비슷할 수 있기때문에, 자전거를 이용할수도 있고, 안할 수도 있다. 때문에 습도와 함께 묶음으로써, 예측력을 높일 수 있을 것이다.
2. 온도가 높던 낮던 비나 눈만 오지 않는다면 출퇴근을 목적으로 사용하는 사람들은 불가피하게 사용할 수도 있다.
3. 온도만으로는 정확한 지표가 될 수 없다고 생각한다. 온도가 높을수록 많이 이용하는 것도 아니고, 온도가 낮을수록 적게 이용하는 것도 아닐 것이기 때문이다.

### temp 컬럼 시각화하기


```python
figure, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)
figure.set_size_inches(18, 8)

sns.barplot(data=train, x="atemp", y="count", ax=ax1)
sns.barplot(data=train, x="temp", y="count", ax=ax2)
sns.barplot(data=train, x="temp", y="atemp", ax=ax3)
sns.barplot(data=train, x="temp", y="humidity", ax=ax4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7bf78a790>




![png](output_49_1.png)



```python
train[["temp", "atemp"]].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp</th>
      <th>atemp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp</th>
      <td>1.000000</td>
      <td>0.984961</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.984961</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
figure, (ax1, ax2) = plt.subplots(nrows = 2,)
figure.set_size_inches(18, 8)

sns.barplot(data=test, x="temp", y="atemp", ax=ax1)
sns.barplot(data=test, x="temp", y="humidity", ax=ax2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7c076b850>




![png](output_51_1.png)



```python
plt.figure(figsize = (18, 4))
sns.scatterplot(data=train, x="temp", y="count", hue="season")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7a47ce160>




![png](output_52_1.png)



```python
plt.figure(figsize = (18, 4))
sns.lmplot(data=train, x="temp", y="count", hue="season")
```




    <seaborn.axisgrid.FacetGrid at 0x7fb7a37c1220>




    <Figure size 1296x288 with 0 Axes>



![png](output_53_2.png)



```python
pd.pivot_table(train, index="temp", values="count", aggfunc=("count","mean"))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>temp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.82</th>
      <td>7.0</td>
      <td>77.714286</td>
    </tr>
    <tr>
      <th>1.64</th>
      <td>2.0</td>
      <td>91.500000</td>
    </tr>
    <tr>
      <th>2.46</th>
      <td>5.0</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>3.28</th>
      <td>11.0</td>
      <td>19.272727</td>
    </tr>
    <tr>
      <th>4.10</th>
      <td>42.0</td>
      <td>52.309524</td>
    </tr>
    <tr>
      <th>4.92</th>
      <td>60.0</td>
      <td>58.416667</td>
    </tr>
    <tr>
      <th>5.74</th>
      <td>107.0</td>
      <td>53.233645</td>
    </tr>
    <tr>
      <th>6.56</th>
      <td>146.0</td>
      <td>68.109589</td>
    </tr>
    <tr>
      <th>7.38</th>
      <td>105.0</td>
      <td>67.704762</td>
    </tr>
    <tr>
      <th>8.20</th>
      <td>229.0</td>
      <td>81.995633</td>
    </tr>
    <tr>
      <th>9.02</th>
      <td>247.0</td>
      <td>73.477733</td>
    </tr>
    <tr>
      <th>9.84</th>
      <td>294.0</td>
      <td>86.442177</td>
    </tr>
    <tr>
      <th>10.66</th>
      <td>332.0</td>
      <td>92.560241</td>
    </tr>
    <tr>
      <th>11.48</th>
      <td>179.0</td>
      <td>110.195531</td>
    </tr>
    <tr>
      <th>12.30</th>
      <td>384.0</td>
      <td>120.302083</td>
    </tr>
    <tr>
      <th>13.12</th>
      <td>356.0</td>
      <td>148.547753</td>
    </tr>
    <tr>
      <th>13.94</th>
      <td>412.0</td>
      <td>145.296117</td>
    </tr>
    <tr>
      <th>14.76</th>
      <td>467.0</td>
      <td>152.957173</td>
    </tr>
    <tr>
      <th>15.58</th>
      <td>255.0</td>
      <td>179.682353</td>
    </tr>
    <tr>
      <th>16.40</th>
      <td>399.0</td>
      <td>170.506266</td>
    </tr>
    <tr>
      <th>17.22</th>
      <td>356.0</td>
      <td>182.609551</td>
    </tr>
    <tr>
      <th>18.04</th>
      <td>327.0</td>
      <td>160.847095</td>
    </tr>
    <tr>
      <th>18.86</th>
      <td>405.0</td>
      <td>159.602469</td>
    </tr>
    <tr>
      <th>19.68</th>
      <td>170.0</td>
      <td>185.058824</td>
    </tr>
    <tr>
      <th>20.50</th>
      <td>327.0</td>
      <td>204.672783</td>
    </tr>
    <tr>
      <th>21.32</th>
      <td>362.0</td>
      <td>196.480663</td>
    </tr>
    <tr>
      <th>22.14</th>
      <td>402.0</td>
      <td>184.480100</td>
    </tr>
    <tr>
      <th>22.96</th>
      <td>394.0</td>
      <td>212.703046</td>
    </tr>
    <tr>
      <th>23.78</th>
      <td>203.0</td>
      <td>235.650246</td>
    </tr>
    <tr>
      <th>24.60</th>
      <td>390.0</td>
      <td>237.182051</td>
    </tr>
    <tr>
      <th>25.42</th>
      <td>402.0</td>
      <td>221.129353</td>
    </tr>
    <tr>
      <th>26.24</th>
      <td>452.0</td>
      <td>232.508850</td>
    </tr>
    <tr>
      <th>27.06</th>
      <td>394.0</td>
      <td>211.025381</td>
    </tr>
    <tr>
      <th>27.88</th>
      <td>224.0</td>
      <td>203.433036</td>
    </tr>
    <tr>
      <th>28.70</th>
      <td>427.0</td>
      <td>257.679157</td>
    </tr>
    <tr>
      <th>29.52</th>
      <td>353.0</td>
      <td>277.691218</td>
    </tr>
    <tr>
      <th>30.34</th>
      <td>299.0</td>
      <td>303.193980</td>
    </tr>
    <tr>
      <th>31.16</th>
      <td>242.0</td>
      <td>352.801653</td>
    </tr>
    <tr>
      <th>31.98</th>
      <td>98.0</td>
      <td>318.683673</td>
    </tr>
    <tr>
      <th>32.80</th>
      <td>200.0</td>
      <td>356.485000</td>
    </tr>
    <tr>
      <th>33.62</th>
      <td>130.0</td>
      <td>348.323077</td>
    </tr>
    <tr>
      <th>34.44</th>
      <td>80.0</td>
      <td>340.225000</td>
    </tr>
    <tr>
      <th>35.26</th>
      <td>76.0</td>
      <td>342.934211</td>
    </tr>
    <tr>
      <th>36.08</th>
      <td>23.0</td>
      <td>362.869565</td>
    </tr>
    <tr>
      <th>36.90</th>
      <td>46.0</td>
      <td>318.717391</td>
    </tr>
    <tr>
      <th>37.72</th>
      <td>34.0</td>
      <td>332.176471</td>
    </tr>
    <tr>
      <th>38.54</th>
      <td>7.0</td>
      <td>238.857143</td>
    </tr>
    <tr>
      <th>39.36</th>
      <td>6.0</td>
      <td>317.833333</td>
    </tr>
    <tr>
      <th>41.00</th>
      <td>1.0</td>
      <td>294.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=train, x="temp", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7a37c67f0>




![png](output_55_1.png)


### Q4-2) 이 시각화로 발견할 수 있는 사실은 어떤 게 있을까요? 그리고 앞서 우리의 예상과 어떤 차이가 있나요?

1. 전반적으로 온도가 높을수록 이용률이 높아진다. 예측과는 다른 추이이다. 온도가 높을수록 비가 오지않는 등 자전거를 탈 여건이 충족되어서 그럴 수도 있다.
2. 하지만 온도가 너무 낮을 때와 온도가 너무 높을 때는 일반적인 추이와 다르다. 피벗 테이블을 살펴본 결과, 모수가 너무 적다고 판단된다.
3. atemp는 temp와 거의 비례한다. season에 따라 temp의 분포가 나타난다. season이 1일때가 가장 낮고, 2,3일때가 가장 높다. 온도가 높을수록 humidity가 증가하다가 특정 온도 이상이되면 온도가 높아지면 humidity가 낮아진다. 

### Q4-3) 이 사실을 통해 어떻게 예측 모델을 개선할 수 있을까요? 최소 3가지 아이디어를 내보세요.
1. 온도가 너무 높을 때와 너무 낮을 때는 모수도 부족하고, temp와 count의 일반적인 관계와 다르다. 표본의 수가 10 이하인 데이터는 지우기로 한다.
2. temp의 수치가 높을수록 count가 높아진다. 다른 변수와의 관계를 이용할 수 있을 것이다. 이를 테면, 불쾌지수 데이터를 만들어 볼 수 있다.
3. atemp가 temp와 거의 유사하다. (cor값이 0.98이상이다) (두 변수는 다중공선성을 띈다)


```python
# 1에서 언급한 표본의 모수 10 이하 데이터 지우기
train = train.loc[(train["temp"] < 38) & (train["temp"] > 3)]

print(train.shape)
train.loc[train["temp"] > 38]
```

    (10841, 21)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>...</th>
      <th>count</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 21 columns</p>
</div>



## 나머지 데이터를 시각화를 통해 더 분석하기

지금까지 분석한 결과 외에도 다양한 방식으로 데이터를 분석하거나 시각화하여 데이터를 더 깊게 이해할려는 시도를 할 수 있습니다. 아직 우리는 분석하지 않은 다양한 데이터(```season```, ```holiday```, ```workingday```, ```humidity```, etc)가 있으며, 이 데이터에서 우리가 기존에 발견하지 못한 (내지는 머신러닝도 스스로 발견하지 못하는) 힌트를 발견할 수 있습니다.

몇몇 힌트를 드리자면

  * 체감 온도(```atemp```)라는게 구체적으로 어떤 개념인지 한 번 고민해보세요. 체감온도를 측정하기 위해서 자전거를 대여하는 사람의 몸에다가 일일이 센서를 붙일 수 없습니다. 분명 다른 방식으로 체감 온도를 측정하거나 계산하고 있을 것입니다.
  * 또한 비슷하게, 비회원(```casual```)과 회원(```registered```)이 어떤 의미인지 한 번 고민해보세요. 일반적으로 자전거 대여량을 측정할 때 이렇게 디테일하게 측정하지 않을 것입니다. (=분명 다른 이유가 있기 때문에 이런 방식으로 측정할 것 같습니다)
  * 그리고 위 컬럼이 아닌, 완전 새로운 개념에 해당하는 컬럼을 추가한 뒤 이를 feature로 사용하는 것도 가능합니다. 가령 1) 날짜 데이터를 갖고 있다면 우리는 요일(dayofweek) 정보를 뽑아낼 수 있고, 2) 온도(```temp```)와 습도(```humidity```)를 알고 있다면 우리는 불쾌지수(discomfort index)를 계산할 수 있습니다. 이러한 정보들을 머신러닝 알고리즘에 적용하면 머신러닝이 새로운 정보를 알 수 있을 것입니다.
  * 다만 데이터를 분석하거나 시각화 할 때, 처음에는 label(맞춰야 하는 정답)을 기준으로 분석하는 것이 효율적이라는 점을 유의해주세요. 시각화를 할 때도 x, y, hue 중에 가능한 한 축을 ```count```컬럼으로 놓고 분석하는 것이 유리합니다.
  * 그리고 비슷한 이유로, 다른 모든 컬럼보다 ```count```컬럼을 완벽하게 분석하고 이해하는 것이 중요합니다. ```count```의 전반적인 분포와 최소/최대치, 그리고 ```casual```과 ```registered```의 관계 등을 집중적으로 분석해주세요.
  
  
위의 힌트, 또는 본인이 생각하기에 중요하다고 생각되는 부분을 분석해보세요. 주어진 형식에 구애받지 않고 자유롭게 데이터를 분석하면 됩니다. 하지만 분석에 과정에서 몇몇 도움이 되는 노하우를 공유하자면

  * 위의 힌트를 포함한 대부분은 구글에서 검색하면 쉽게 찾을 수 있습니다. 가령 1) 체감 온도(```atemp```)의 개념과 이를 측정 또는 계산하는 방식, 2) 판다스(Pandas)를 활용해 날짜 데이터에서 요일(dayofweek) 정보를 뽑는 법 등등. 대부분의 노하우들은 인터넷에 이미 존재합니다. 이를 빠르게 검색해서 내 코드에 적용하는 것도 데이터 사이언티스트들의 중요한 소양이자 실력입니다.
  * 정보를 얻을 때, 창의성도 중요하지만 유사 솔루션, 경진대회, 데이터셋을 벤치마킹하는 실력도 매우 중요합니다. 캐글에서는 보통 [Kernel](https://www.kaggle.com/c/bike-sharing-demand/kernels) 탭에서 사람들이 본인들만의 분석 결과와 솔루션을 올리고, [Discussion](https://www.kaggle.com/c/bike-sharing-demand/discussion) 탭에서 경진대회에 대한 토론을 합니다. 이 탭을 집중적으로 살펴보고 벤치마킹 해주세요. 심지어 [이런](https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile/notebook) 페이지에는 경진대회 상위 10%에 도달하는 노하우가 그대로 공유되어 있습니다. 이 노하우만 잘 이해해도 충분합니다.
  * 그리고 비슷하게, [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) 경진대회의 다양한 솔루션들을 구글에서 찾을 수도 있습니다. [다음의 링크](https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/)나 [다음의 링크](https://medium.com/@viveksrinivasan/how-to-finish-top-10-percentile-in-bike-sharing-demand-competition-in-kaggle-part-1-c816ea9c51e1)처럼 이 경진대회에 대해 자세히 분석하고 솔루션을 제시하는 곳도 있습니다. 이런 솔루션을 구글에서 찾아서 적극적으로 벤치마킹 해주세요.
  * 마지막으로, 데이터는 많이 분석하면 분석할수록 노하우가 쌓입니다. 그리고 캐글 경진대회도 많이 참여할수록 점점 노하우가 쌓이게 됩니다. 그런 의미에서, 이전에 참여한 경진대회에서 먹혔던 분석 노하우가 전략을 적극적으로 활용해보세요. 가령 [Titanic](https://www.kaggle.com/c/titanic) 경진대회에서 먹혔던 전략을 그대로 활용하는 것도 가능합니다.
  

---
### 요일 데이터 dayofweek 생성


```python
train["datetime-weekday"] = train["datetime"].dt.weekday

print(train.shape)
train["datetime-weekday"].head()
```

    (10841, 22)





    0    5
    1    5
    2    5
    3    5
    4    5
    Name: datetime-weekday, dtype: int64




```python
train.loc[train["datetime-weekday"] == 0, "dayofweek"] = "Monday"
train.loc[train["datetime-weekday"] == 1, "dayofweek"] = "Tuesday"
train.loc[train["datetime-weekday"] == 2, "dayofweek"] = "Wednesday"
train.loc[train["datetime-weekday"] == 3, "dayofweek"] = "Thursday"
train.loc[train["datetime-weekday"] == 4, "dayofweek"] = "Friday"
train.loc[train["datetime-weekday"] == 5, "dayofweek"] = "Saturday"
train.loc[train["datetime-weekday"] == 6, "dayofweek"] = "Sunday"

print(train.shape)
train.head()
```

    (10841, 23)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>...</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 05:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>9.84</td>
      <td>12.880</td>
      <td>75</td>
      <td>6.0032</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 10:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>15.58</td>
      <td>19.695</td>
      <td>76</td>
      <td>16.9979</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 11:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>14.76</td>
      <td>16.665</td>
      <td>81</td>
      <td>19.0012</td>
      <td>26</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 12:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17.22</td>
      <td>21.210</td>
      <td>77</td>
      <td>19.0012</td>
      <td>29</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 13:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18.86</td>
      <td>22.725</td>
      <td>72</td>
      <td>19.9995</td>
      <td>47</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
test["datetime-weekday"] = test["datetime"].dt.weekday

print(test.shape)
test["datetime-weekday"].head()
```

    (6493, 19)





    0    3
    1    3
    2    3
    3    3
    4    3
    Name: datetime-weekday, dtype: int64




```python
test.loc[test["datetime-weekday"] == 0, "dayofweek"] = "Monday"
test.loc[test["datetime-weekday"] == 1, "dayofweek"] = "Tuesday"
test.loc[test["datetime-weekday"] == 2, "dayofweek"] = "Wednesday"
test.loc[test["datetime-weekday"] == 3, "dayofweek"] = "Thursday"
test.loc[test["datetime-weekday"] == 4, "dayofweek"] = "Friday"
test.loc[test["datetime-weekday"] == 5, "dayofweek"] = "Saturday"
test.loc[test["datetime-weekday"] == 6, "dayofweek"] = "Sunday"

print(test.shape)
test.head()
```

    (6493, 20)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
  </tbody>
</table>
</div>



### 시각화 전 생각해보기
1. 주말에는 이용률이 평일보다 적을 것이다. (출퇴근하는 사람들이 줄어들기 때문)
2. 요일별로 시간별 이용률이 다르게 나타날 것이다.
3. 요일별 횟수 차이는 크지 않을 것이다.


```python
sns.barplot(data=train, x="datetime-weekday", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7c1c69c40>




![png](output_66_1.png)



```python
sns.pointplot(data=train, x="datetime-hour", y="count", hue="datetime-weekday")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7a56d8ac0>




![png](output_67_1.png)



```python
pd.pivot_table(train, index=["datetime-weekday", "dayofweek"], values="count", aggfunc=["count", "mean", "min", "max"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>Monday</th>
      <td>1547</td>
      <td>190.372334</td>
      <td>1</td>
      <td>968</td>
    </tr>
    <tr>
      <th>1</th>
      <th>Tuesday</th>
      <td>1535</td>
      <td>190.026059</td>
      <td>1</td>
      <td>970</td>
    </tr>
    <tr>
      <th>2</th>
      <th>Wednesday</th>
      <td>1535</td>
      <td>189.518567</td>
      <td>1</td>
      <td>977</td>
    </tr>
    <tr>
      <th>3</th>
      <th>Thursday</th>
      <td>1550</td>
      <td>197.106452</td>
      <td>1</td>
      <td>901</td>
    </tr>
    <tr>
      <th>4</th>
      <th>Friday</th>
      <td>1529</td>
      <td>197.844343</td>
      <td>1</td>
      <td>900</td>
    </tr>
    <tr>
      <th>5</th>
      <th>Saturday</th>
      <td>1572</td>
      <td>196.218193</td>
      <td>1</td>
      <td>783</td>
    </tr>
    <tr>
      <th>6</th>
      <th>Sunday</th>
      <td>1573</td>
      <td>180.764781</td>
      <td>1</td>
      <td>757</td>
    </tr>
  </tbody>
</table>
</div>



### 시각화 후 알게된 것
1. 예측과 달리 토요일의 이용률은 목요일과 금요일과 비슷한 수준이다. (월화수, 수목금, 일) 세 그룹으로 나뉜다.
2. 일요일의 이용률이 가장 적었다.
3. 주말과 평일엔 시간대별 이용률이 확연히 구분되며, 서로 비슷한 그래프를 보인다. 토요일은 주말의 모습을 보여주지만, 일요일보다는 이용률이 훨씬 높은 것으로 나타난다.

### 분석에 활용
1. 범주형 데이터로써, One Hot Encoding을 해야 할 것이다.
2. 시간대별 그래프를 보면, 주말과 평일을 나누어 묶을 수 있을 것처럼 보이지만, 피벗 테이블을 살펴봤을 때, 그러지 못할 것으로 생각된다.


```python
# 1에서 언급한 one hot encoding
train = pd.concat([pd.get_dummies(train.dayofweek), train], axis=1)
print(train.shape)
train.head()
```

    (10841, 30)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>...</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 05:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 10:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 11:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 12:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 13:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
test = pd.concat([pd.get_dummies(test.dayofweek), test], axis=1)
print(test.shape)
test.head()
```

    (6493, 27)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>...</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



---
### atemp

국제적으로 atemp를 계산하는 방법은 temp와 windspeed를 활용한다.<br>
atemp = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16<br>
시각화결과, temp과 거의 유사하다.


```python
sns.lmplot(data=train, x="temp", y="atemp")
```




    <seaborn.axisgrid.FacetGrid at 0x7fb7a44dadf0>




![png](output_75_1.png)


---

### season

계절을 가리키며, 봄(1), 여름(2), 가을(3), 겨울(4)로 이루어져있다

### 시각화 이전에 생각해보기
1. temp에서 살펴보았을 때, 봄(1)에 이용률 낮고, [여름(2), 가을(3)]에 높았다.
2. 계절은 숫자 데이터이지만, 범주형 데이터로써 One Hot Encoding을 사용해야 할 것이다.
3. 데이터의 모수가 부족한 곳은 없을 것이라고 추측된다.


```python
pd.pivot_table(train, index="season", values="count", aggfunc=["count","mean"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>season</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2658</td>
      <td>116.492852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2731</td>
      <td>215.268400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2718</td>
      <td>234.125828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2734</td>
      <td>198.988296</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=train, x="season", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7a3f54c40>




![png](output_81_1.png)



```python
sns.scatterplot(data=train, x="season", y="temp")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7ae6560d0>




![png](output_82_1.png)



```python
train[(train["season"]==4)&(train["datetime-year"]==2012)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>...</th>
      <th>datetime-month</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8375</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-10-01 00:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>8376</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-10-01 01:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>8377</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-10-01 04:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>8378</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-10-01 06:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>8379</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-10-01 07:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10881</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-12-17 12:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>17</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>10882</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-12-17 15:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>17</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>10883</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2012-12-18 08:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>18</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>10884</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2012-12-18 22:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>18</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>10885</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2012-12-19 00:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>Wednesday</td>
    </tr>
  </tbody>
</table>
<p>1367 rows × 30 columns</p>
</div>



### 시각화 후 알게 된 것
1. 봄에 이용률이 가장 적고, 가을에 이용률이 가장 높다. 특히 봄의 이용률은 다른 계절보다 절반가량 밖에 되지 않는다.
2. 봄은 1-3월, 여름은 4-6월, 가을은 7-9월, 겨울은 10-12월이다.
3. 가을(3) > 여름(2) > 겨울(4) > 봄(1) 순이다.

### 분석에 활용
1. 범주형 데이터로써, One hot Coding을 통해 컬럼을 새로 만든다.
2. 봄에는 특히 이용률이 훨씬 적다. 따라서 Spring과 Other seasons로 컬럼을 두 개로 나누어볼 수 있다.
3. 혹은 1,4를 묶고, 2,3을 묶어서 분류하는 방법도 존재할 것이다.(추운 계절과 따뜻한 계절)


```python
# 1에서 언급한 대로 One Hot Encoding하기
train.loc[train["season"]==1, "season_name"] = "Spring"
train.loc[train["season"]==2, "season_name"] = "Summer"
train.loc[train["season"]==3, "season_name"] = "Fall"
train.loc[train["season"]==4, "season_name"] = "Winter"

train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>...</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
      <th>season_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 05:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 10:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 11:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 12:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-01 13:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
train = pd.concat([pd.get_dummies(train.season_name), train], axis=1)

print(train.shape)
train.head()
```

    (10841, 35)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fall</th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Winter</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>...</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
      <th>season_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>Saturday</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
test.loc[test["season"]==1, "season_name"] = "Spring"
test.loc[test["season"]==2, "season_name"] = "Summer"
test.loc[test["season"]==3, "season_name"] = "Fall"
test.loc[test["season"]==4, "season_name"] = "Winter"

test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>...</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
      <th>season_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
test = pd.concat([pd.get_dummies(test.season_name), test], axis=1)
print(test.shape)
test.head()
```

    (6493, 32)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fall</th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Winter</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>...</th>
      <th>datetime-day</th>
      <th>datetime-hour</th>
      <th>datetime-minute</th>
      <th>datetime-second</th>
      <th>weather_Clear</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>datetime-weekday</th>
      <th>dayofweek</th>
      <th>season_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>Thursday</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



### holiday

### 시각화 전에 생각해보기
1. datetime에서 출퇴근 시간에 이용률이 높았던 것으로 보아, 출퇴근에 활용하는 사람이 많다고 판단되며, 휴일에는 이용률이 적을 것으로 예상된다.
2. 0과 1로 휴일이다 아니다로 범주형 데이터이다.
3. workingday랑 유사할 것으로 예상된다.


```python
pd.pivot_table(train, index="holiday", values="count", aggfunc=["count","mean","min","max"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>holiday</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10530</td>
      <td>191.843875</td>
      <td>1</td>
      <td>977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>311</td>
      <td>185.877814</td>
      <td>1</td>
      <td>712</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pointplot(data=train, x="datetime-hour", y="count", hue="workingday")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7af203910>




![png](output_93_1.png)



```python
sns.scatterplot(data=train, x="holiday", y="workingday")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7af4f7340>




![png](output_94_1.png)



```python
print(train[(train["holiday"]==0)&(train["workingday"]==0)]["count"].mean(),
train[(train["holiday"]==1)&(train["workingday"]==0)]["count"].mean(),
train[(train["holiday"]==0)&(train["workingday"]==1)]["count"].mean())
```

    188.48903020667726 185.87781350482314 193.27257955314826


### 시각화 후 알게된 것
1. 예상과 달리, 휴일에도 이용률이 크게 낮은 편은 아니다. 휴일이 아닌 날과 비슷한 수준이다.
2. 휴일인 경우는 적다.
3. 휴일이 아니더라도 일을 하지 않는 때가 있는데, 엑셀로 데이터를 살펴본 결과 주말을 의미한다.

### 분석에 활용
1. holiday는 0과 1로 범주형 데이터로 활용한다.
2. 휴일이면서 일하지않는날, 휴일이 아니고 일하는날, 휴일이 아닌데 일하지않는날 비교해보니 큰 차이는 있지 않았다. 다만, 휴일이 아니면서 일하는날에 이용률이 가장 높고, 휴일이면서 일하지 않는날이 이용률이 가장 높았다.
3. holiday와 workingday를 함께 분석에 사용하도록 한다.

---
### workingday

### 시각화전에 생각해보기
1. datetime에서 살펴본 것처럼, 출퇴근에 많고 적은 것으로 보아 일하는 날에 이용률이 더 높을 것으로 예상된다.
2. 0과 1로 일하는 날이 아니다, 맞다로 범주형 데이터이다.
3. holiday랑 유사할 것으로 생각된다.


```python
pd.pivot_table(train, index="workingday", values="count", aggfunc=["count", "mean", "max", "min"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>workingday</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3456</td>
      <td>188.254051</td>
      <td>783</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7385</td>
      <td>193.272580</td>
      <td>977</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot(data=train, x="workingday", y="count")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7c0ddc4f0>




![png](output_101_1.png)


### 시각화 후 알게 된 것
1. 일하는 날에 이용률이 평균적으로 높은 것을 알 수 있다.
2. 일하는 날이 일하지 않는 날보다 두배 이상 많다.

### 분석에 활용
1. 일하는 날이 이용률에 긍정적인 영향을 미친다.
2. holiday와 함께 활용이 가능하다.

---
### humidity

### 시각화 전에 생각해보기
1. 습도가 높을수록 비가 올 확률이 높기 때문에, 이용률이 낮을 것이다.
2. 반대로 습도가 낮아도 너무 건조해 온도가 낮을 수도 있어서, 이용률이 낮을 것이다.
3. 온도(temp)와 함께 불쾌지수로 활용해 볼 수도 있을 것이다.<br>
**불쾌지수(Discomfort Index)** = 1.8 × 온도 - 0.55 (1 - 습도) (1.8 × 온도 - 26) + 32


```python
train["humidity"].describe()
```




    count    10841.000000
    mean        61.990868
    std         19.206338
    min          0.000000
    25%         47.000000
    50%         62.000000
    75%         77.000000
    max        100.000000
    Name: humidity, dtype: float64




```python
sns.lmplot(data=train, x="humidity", y="count", hue="season")
```




    <seaborn.axisgrid.FacetGrid at 0x7fb7c047af10>




![png](output_107_1.png)


### 시각화 후 알게된 정보
1. 대체적으로 습도가 높아질수록 이용률이 낮아진다.
2. 계절별로 습도의 분포가 나타난다. 하지만 또렷한 분포는 아니다.
3. 정확한 구분은 어렵기 때문에, 다른 인덱스와 같이 사용해 볼 수 있을 것이다.

### 분석에 활용
1. 습도가 높아질수록 이용률이 낮아진다는 사실은 맞지만, 분석에 활용할 만큼 큰 차이는 없는 것으로 보인다.
2. 따라서 다른 컬럼과 함께 사용하여 새로운 컬럼을 만들 필요가 있다.
3. 불쾌지수(Discomfort Index)를 만들기로 한다.

---
## 새로운 컬럼
### discomfort

temp와 humidity를 활용하여 불쾌지수 인덱스를 생성하면, 이용률 변화 추이를 잘 살펴볼 수 있을 것으로 판단되었다.<br>
**불쾌지수 인덱스(discomfort)** = 1.8 × temp - 0.55 (1 - 0.01*humidity) (1.8 × temp - 26) + 32<br>
공식을 활용하여 새로운 컬럼(discomfort)에 값을 추가하여 분석에 활용하고자 한다.
불쾌지수는
* 80이상, 매우 높음 (전원 불쾌감을 느낌)
* 75-80, 높음 (불쾌감을 나타내기 시작함)
* 68-75, 보통 (50%정도 불쾌감을 느낌)
* 68미만, 낮음 (전원 쾌적함을 느낌)

로 구분된다.


```python
train["discomfort"] = 1.8*train["temp"] - 0.55*(1 - 0.01*train["humidity"])*(1.8*train["temp"] - 26) + 32

print(train["discomfort"].shape)
train["discomfort"].describe()
```

    (10841,)





    count    10841.000000
    mean        66.149986
    std         10.732458
    min         40.225088
    25%         57.326718
    50%         66.231864
    75%         75.611944
    max         86.531176
    Name: discomfort, dtype: float64




```python
test["discomfort"] = 1.8*test["temp"] - 0.55*(1 - 0.01*test["humidity"])*(1.8*test["temp"] - 26) + 32

print(test["discomfort"].shape)
test["discomfort"].describe()
```

    (6493,)





    count    6493.000000
    mean       66.797760
    std        11.255488
    min        38.601516
    25%        57.126958
    50%        67.585212
    75%        76.927360
    max        90.021472
    Name: discomfort, dtype: float64




```python
plt.figure(figsize=(18,4))

sns.lmplot(data=train, x="discomfort", y="count", hue="season")
```




    <seaborn.axisgrid.FacetGrid at 0x7fb7af203d90>




    <Figure size 1296x288 with 0 Axes>



![png](output_113_2.png)



```python
train[["temp", "atemp", "humidity", "datetime-month", "datetime-hour", "windspeed", "discomfort"]].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>datetime-month</th>
      <th>datetime-hour</th>
      <th>windspeed</th>
      <th>discomfort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp</th>
      <td>1.000000</td>
      <td>0.984777</td>
      <td>-0.064626</td>
      <td>0.254019</td>
      <td>0.142186</td>
      <td>0.004709</td>
      <td>0.986948</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.984777</td>
      <td>1.000000</td>
      <td>-0.043464</td>
      <td>0.260397</td>
      <td>0.136999</td>
      <td>-0.031605</td>
      <td>0.973580</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>-0.064626</td>
      <td>-0.043464</td>
      <td>1.000000</td>
      <td>0.201732</td>
      <td>-0.279668</td>
      <td>-0.341358</td>
      <td>0.034781</td>
    </tr>
    <tr>
      <th>datetime-month</th>
      <td>0.254019</td>
      <td>0.260397</td>
      <td>0.201732</td>
      <td>1.000000</td>
      <td>-0.008629</td>
      <td>-0.139636</td>
      <td>0.269952</td>
    </tr>
    <tr>
      <th>datetime-hour</th>
      <td>0.142186</td>
      <td>0.136999</td>
      <td>-0.279668</td>
      <td>-0.008629</td>
      <td>1.000000</td>
      <td>0.146676</td>
      <td>0.113274</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.004709</td>
      <td>-0.031605</td>
      <td>-0.341358</td>
      <td>-0.139636</td>
      <td>0.146676</td>
      <td>1.000000</td>
      <td>-0.018360</td>
    </tr>
    <tr>
      <th>discomfort</th>
      <td>0.986948</td>
      <td>0.973580</td>
      <td>0.034781</td>
      <td>0.269952</td>
      <td>0.113274</td>
      <td>-0.018360</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_data = train[["temp", "atemp", "humidity", "datetime-month", "datetime-hour", "windspeed", "discomfort"]]

colormap = plt.cm.PuBu

f, ax = plt.subplots(figsize = (12,10))
plt.title('Correlation Analysis', y=1, size=18)
sns.heatmap(corr_data.corr(), vmax=.8, linewidths=0.1, square=True, annot=True, cmap=colormap, linecolor="white", annot_kws={'size':14})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7c1f5d850>




![png](output_115_1.png)



```python
fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4)
fig.set_size_inches(12,8)

sns.regplot(data=train, x="temp", y="count", ax=ax1)
sns.regplot(data=train, x="atemp", y="count", ax=ax2)
sns.regplot(data=train, x="humidity", y="count", ax=ax3)
sns.regplot(data=train, x="discomfort", y="count", ax=ax4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb7a832a130>




![png](output_116_1.png)


### 시각화 후
1. discomfort는 temp와 atemp와 다중공선성을 띈다.
2. 불쾌지수가 높을수록 이용률이 높다는 것을 알 수 있었다.

---
### casual, registered 살펴보기

count = casual + registered 이다.

예상컨데, 출퇴근용으로 사용하는 사람들은 registered로 등록되어 있을 가능성이 높다.
또한, 휴일에 사용하는 사람들은 정기적으로 사용하는 사람들이 아니기 때문에 casual로 등록될 가능성이 있다.
그리고 날씨가 나쁜 날에는 급하게 빌리는 사람이 있어 casual로 등록된 경우가 많을 것이다.

따라서,
1. 출퇴근 시간과 일하는 날에는 registered의 비율이 더 높을 것이다.
2. 휴일과 일하지 않는 날에는 casual의 비율이 평소보다 높을 것이다.
3. 날씨가 안좋은 날일 수록 casual의 비율이 평소보다 높을 것이다.

이를 살펴보기 위해서, 비회원 비율(casual_rate)컬럼을 만들기로 한다.
casual_rate는 %가 높을수록, 비회원이 많다는 것을 의미한다.
또한, casual_rate가 낮을수록, 회원이 많다는 것을 의미한다.

1. 출퇴근 시간과 일하는 날에는 casual_rate가 낮을 것이다.
2. 휴일과 일하지 않는 날에는 casual_rate가 높을 것이다.
3. 날씨가 안좋은 날일 수록 casual_rate가 높을 것이다.


```python
train[["casual","registered","count"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>24</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>30</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>55</td>
      <td>84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>47</td>
      <td>94</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10881</th>
      <td>21</td>
      <td>211</td>
      <td>232</td>
    </tr>
    <tr>
      <th>10882</th>
      <td>15</td>
      <td>196</td>
      <td>211</td>
    </tr>
    <tr>
      <th>10883</th>
      <td>10</td>
      <td>652</td>
      <td>662</td>
    </tr>
    <tr>
      <th>10884</th>
      <td>5</td>
      <td>127</td>
      <td>132</td>
    </tr>
    <tr>
      <th>10885</th>
      <td>6</td>
      <td>35</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>10841 rows × 3 columns</p>
</div>




```python
train["casual_rate"] = (train["casual"] / train["count"]) * 100

train["casual_rate"]
```




    0         0.000000
    1        33.333333
    2        46.428571
    3        34.523810
    4        50.000000
               ...    
    10881     9.051724
    10882     7.109005
    10883     1.510574
    10884     3.787879
    10885    14.634146
    Name: casual_rate, Length: 10841, dtype: float64




```python
train["casual_rate"].describe()
```




    count    10841.000000
    mean        17.097695
    std         13.676788
    min          0.000000
    25%          6.194690
    50%         14.473684
    75%         25.287356
    max        100.000000
    Name: casual_rate, dtype: float64




```python
figure, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
figure.set_size_inches(12,10)

sns.barplot(data=train, x="holiday", y="casual_rate", ax=ax1)
sns.barplot(data=train, x="workingday", y="casual_rate", ax=ax2)
sns.barplot(data=train, x="datetime-hour", y="casual_rate", ax=ax3)
sns.barplot(data=train, x="weather", y="casual_rate", ax=ax4)
sns.barplot(data=train, x="datetime-weekday", y="casual_rate", ax=ax5)

ax1.set(title="holiday")
ax2.set(title="workingday")
ax3.set(title="datetime-hour")
ax4.set(title="weather")
ax5.set(title="datetime-weekday")
```




    [Text(0.5, 1.0, 'datetime-weekday')]




![png](output_122_1.png)


### 시각화 이후
0. 전체적으로 회원의 비율이 높은 기업이다.
1. 휴일일 경우 비회원 비율이 높다. 예상과 일치
2. 일하는 날에는 회원 비율이 높다. 예상과 일치
3. 출퇴근 시간에는 회원 비율이 높다. 예상과 일치
4. 날씨가 나빠질수록 회원 비율이 높다. 날씨가 나빠지면, 자전거 회원이 더 많이 이용한다. (날씨 4는 데이터가 1개이기 때문에 제외한다.)
5. 주말(5,6)에는 비회원 비율이 높다.

## 수고 많으셨습니다!


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10841 entries, 0 to 10885
    Data columns (total 36 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   Fall              10841 non-null  uint8         
     1   Spring            10841 non-null  uint8         
     2   Summer            10841 non-null  uint8         
     3   Winter            10841 non-null  uint8         
     4   Friday            10841 non-null  uint8         
     5   Monday            10841 non-null  uint8         
     6   Saturday          10841 non-null  uint8         
     7   Sunday            10841 non-null  uint8         
     8   Thursday          10841 non-null  uint8         
     9   Tuesday           10841 non-null  uint8         
     10  Wednesday         10841 non-null  uint8         
     11  datetime          10841 non-null  datetime64[ns]
     12  season            10841 non-null  int64         
     13  holiday           10841 non-null  int64         
     14  workingday        10841 non-null  int64         
     15  weather           10841 non-null  int64         
     16  temp              10841 non-null  float64       
     17  atemp             10841 non-null  float64       
     18  humidity          10841 non-null  int64         
     19  windspeed         10841 non-null  float64       
     20  casual            10841 non-null  int64         
     21  registered        10841 non-null  int64         
     22  count             10841 non-null  int64         
     23  datetime-year     10841 non-null  int64         
     24  datetime-month    10841 non-null  int64         
     25  datetime-day      10841 non-null  int64         
     26  datetime-hour     10841 non-null  int64         
     27  datetime-minute   10841 non-null  int64         
     28  datetime-second   10841 non-null  int64         
     29  weather_Clear     10841 non-null  bool          
     30  weather_Cloudy    10841 non-null  bool          
     31  weather_Bad       10841 non-null  bool          
     32  datetime-weekday  10841 non-null  int64         
     33  dayofweek         10841 non-null  object        
     34  season_name       10841 non-null  object        
     35  discomfort        10841 non-null  float64       
    dtypes: bool(3), datetime64[ns](1), float64(4), int64(15), object(2), uint8(11)
    memory usage: 2.0+ MB



```python
feature_names = ["Spring", "Summer", "Fall", "Winter",
                 "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                 "holiday", "workingday", "weather_Clear", "weather_Cloudy", "weather_Bad",
                 "temp", "atemp","humidity", "windspeed", "discomfort",
                 "datetime-year", "datetime-month","datetime-hour"]
feature_names
```




    ['Spring',
     'Summer',
     'Fall',
     'Winter',
     'Monday',
     'Tuesday',
     'Wednesday',
     'Thursday',
     'Friday',
     'Saturday',
     'Sunday',
     'holiday',
     'workingday',
     'weather_Clear',
     'weather_Cloudy',
     'weather_Bad',
     'temp',
     'atemp',
     'humidity',
     'windspeed',
     'discomfort',
     'datetime-year',
     'datetime-month',
     'datetime-hour']




```python
X_train = train[feature_names]

print(X_train.shape)
X_train.head()
```

    (10841, 24)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Fall</th>
      <th>Winter</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
      <th>Saturday</th>
      <th>...</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>discomfort</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>9.84</td>
      <td>12.880</td>
      <td>75</td>
      <td>6.0032</td>
      <td>50.851600</td>
      <td>2011</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>15.58</td>
      <td>19.695</td>
      <td>76</td>
      <td>16.9979</td>
      <td>59.774192</td>
      <td>2011</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>14.76</td>
      <td>16.665</td>
      <td>81</td>
      <td>19.0012</td>
      <td>58.508644</td>
      <td>2011</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>17.22</td>
      <td>21.210</td>
      <td>77</td>
      <td>19.0012</td>
      <td>62.364006</td>
      <td>2011</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>18.86</td>
      <td>22.725</td>
      <td>72</td>
      <td>19.9995</td>
      <td>64.724008</td>
      <td>2011</td>
      <td>1</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
X_test = test[feature_names]

print(X_test.shape)
X_test.head()
```

    (6493, 24)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Fall</th>
      <th>Winter</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
      <th>Saturday</th>
      <th>...</th>
      <th>weather_Cloudy</th>
      <th>weather_Bad</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>discomfort</th>
      <th>datetime-year</th>
      <th>datetime-month</th>
      <th>datetime-hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>52.836504</td>
      <td>2011</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>52.836504</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>52.836504</td>
      <td>2011</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>52.836504</td>
      <td>2011</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>52.836504</td>
      <td>2011</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
label_name = "count"

y_train = train[label_name]

print(y_train.shape)
y_train.head()
```

    (10841,)





    0     1
    1    36
    2    56
    3    84
    4    94
    Name: count, dtype: int64




```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state = 37)
model
```




    RandomForestRegressor(random_state=37)




```python
model.fit(X_train, y_train)
```




    RandomForestRegressor(random_state=37)




```python
prediction_list = model.predict(X_test)

print(prediction_list.shape)
prediction_list
```

    (6493,)





    array([  9.71,   4.96,   4.7 , ..., 139.79, 117.77,  65.38])




```python
submit = pd.read_csv("data/sampleSubmission.csv")

submit["count"] = prediction_list

print(submit.shape)
submit.head()
```

    (6493, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>9.71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>4.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>3.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>2.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit.to_csv("data/RF_01.csv", index=False)
```


```python

```
