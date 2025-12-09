
```
team_1
├─ malware-cnn-project
│  ├─ backend
│  │  ├─ db.sqlite3
│  │  ├─ detection
│  │  │  ├─ management
│  │  │  │  └─ commands
│  │  │  │     ├─ generate_malware_images.py
│  │  │  │     ├─ train_convnext.py
│  │  │  │     └─ train_simple_lstm.py
│  │  │  ├─ ml_models
│  │  │  │  ├─ encoder_lstm.pkl
│  │  │  │  ├─ lstm_model.h5
│  │  │  │  ├─ pytorch
│  │  │  │  │  ├─ final_convnext_malware_weighted_10classes.pth
│  │  │  │  │  └─ my_trained_convnext.pth
│  │  │  │  ├─ scaler_lstm.pkl
│  │  │  │  ├─ tensorflow_keras
│  │  │  │  │  ├─ encoders_map.pkl
│  │  │  │  │  ├─ encoder_lstm.pkl
│  │  │  │  │  ├─ lstm_model.h5
│  │  │  │  │  ├─ scaler_lstm.pkl
│  │  │  │  │  └─ threshold_lstm.txt
│  │  │  │  └─ threshold_lstm.txt
│  │  │  ├─ model_arch.py
│  │  │  ├─ result_images
│  │  │  │  ├─ confusion_matrix_cnn.png
│  │  │  │  ├─ confusion_matrix_rf.png
│  │  │  │  └─ confusion_matrix_rf_smote.png
│  │  │  ├─ templates
│  │  │  │  └─ detection
│  │  │  │     └─ dashboard.html
│  │  │  ├─ urls.py
│  │  │  └─ views.py
│  │  ├─ main
│  │  │  ├─ asgi.py
│  │  │  ├─ settings.py
│  │  │  ├─ urls.py
│  │  │  ├─ wsgi.py
│  │  │  └─ __init__.py
│  │  ├─ make_test_image.py
│  │  ├─ manage.py
│  │  ├─ test_api.py
│  │  ├─ test_neptune.png
│  │  └─ test_pipeline.py
│  ├─ frontend
│  │  └─ templates
│  │     └─ mainpage.html
│  ├─ README.md
│  └─ requirements.txt
└─ README.md

```