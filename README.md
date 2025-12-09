# Ransomware-Behavior-Analytics

## 프로젝트 개요  
본 프로젝트는 현업의 보안 분석 중 행위 특성을 토대로 Ransomware의 Class 분류 모델을 Autoencoder와 LSTM를 활용하여 학습 및 테스트하는 프로젝트입니다.  
실제 악성 코드 바이너리를 다루지 않고, 행위 피처(behavior features)만 통계적으로 묘사한 Synthetic 데이터셋 생성하여 학습 및 테스트하였습니다.

### 주요 기능
- Autoencoder 모델을 통해 Ransomware인지 아닌지 분석
- LSTM 모델을 통해 Ransomware Class 분류

### 기술 스택
| 구성 요소        | 기술 |
|----------------|------|
| 백엔드         | Django, Jupyter|
| 프론트엔드     | HTML, CSS, JavaScript |
| AI      | Autoencoder, LSTM |

### 설치 및 실행 방법
```
git clone https://github.com/minkyungbae/Ransomware-Behavior-Analytics.git
pip install -r requirements.txt
cd backend
python manage.py migrate
python manage.py runserver

```

### 폴더 구조
```
Ransomware-Behavior-Analytics
├─ backend/                           # Backend 코드 관리 폴더
│  ├─ main/                           # Django 프로젝트 관리 폴더
│  │  ├─ asgi.py
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  ├─ wsgi.py
│  │  └─ __init__.py
│  ├─ manage.py
│  ├─ ML/                             # 머신러닝 및 데이터셋 관리 폴더
│  │  ├─ ransomwaredataset.csv
│  │  ├─ TrainML.py                   # 머신러닝 학습 및 테스트 코드 파일
│  │  └─ __init__.py
│  └─ ransomware/                     # Ransomware 작업 관리 폴더(Django App)
│     ├─ admin.py
│     ├─ apps.py
│     ├─ migrations
│     │  └─ __init__.py
│     ├─ models.py
│     ├─ tests.py
│     ├─ urls.py
│     ├─ views.py
│     └─ __init__.py
├─ frontend/                          # 템플릿 관리 폴더
│  └─ mainpage.html
├─ ransomware-model.ipynb             # 머신러닝 학습 및 테스트 원본 코드 파일
├─ README.md
└─ requirements.txt                   # 필요한 모듈 및 라이브러리 파일

```