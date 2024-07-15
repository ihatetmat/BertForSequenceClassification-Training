## BertForSequnceClassification
OCR API 를 통해 추출된 텍스트를 카테고리로 변환하기 위한 모델 훈련

```
── root
   ├── Training_kor.py ( 훈련 코드 )
   ├── Test_kor.ipynb ( 훈련 코드의 결과로 생성된 모댈을 이용한 테스트 )
   └── training_kor.sh ( Background Training, Memory Profiling 을 위한 쉘 파일 )
   ├── result ( 훈련 도중 생성되는 파일 )
   │   ├── label.csv ( 텍스트 카테고리 -> 정수 )
   │   ├── failed_predictions.csv ( 테스트 과정에서 추론에 실패한 데이터 세트 )
   │   └── kor_bert_model_training.log ( 훈련 로그 )
   └── dataset
       └── NaverMarket_Dataset.csv ( 네이버 마켓에서 크롤링한 '제품명 - 식재료 카테고리' 쌍 데이터 세트 )
```

여기서 결과로 생성된 모델이 GCP Storage Bucket 에 저장되고 Recipable-AI 리포지토리의 추론 서버에서 활용됨.
