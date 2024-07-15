#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime
import csv
import logging
import time
import datetime
import os
from google.cloud import storage
LOG_FILENAME = './result/kor_bert_model_training.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
logging.basicConfig(filename=LOG_FILENAME, format=LOG_FORMAT, level=LOG_LEVEL)

df = pd.read_csv('./dataset/NaverMarket_Dataset.csv')

# 수집한 데이터의 클래스를 라벨링 (text -> integer)
label_dict = {}

for idx, label in enumerate(df['label'].unique()):
    label_dict[label] = idx

df['label'] = df['label'].map(label_dict)

logging.info(df)
with open('./result/label.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['label', 'number'])
    for label, number in label_dict.items():
        writer.writerow([label, number])

total_data_count = len(df)

# 데이터를 랜덤하게 섞고 훈련, 테스트 데이터를 분리
df = df.sample(frac=1, random_state=42)
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("훈련 데이터 개수:", len(X_train))
logging.info("테스트 데이터 개수:", len(X_test))

sentences = X_train['product_name']

# BERT의 입력 형식에 맞게 변환 ([CLS] 는 클래스에 대한 정보가 담기는 토큰, [SEP] 은 문장 1과 문장 2를 구분하기 위한 토큰임)
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
labels = y_train.values

# BERT 전용 토크나이저 (한국어 특화된 'kykim/bert-kor-base' 체크포인트를 이용함.)
tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# 입력 토큰의 최대 시퀀스 길이
MAX_LEN = 64

# 토큰을 정수 임베딩으로 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 문장을 MAX_LEN 인 64에 맞추기 위해 padding 추가
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# 어텐션 마스크 (패딩된 토큰의 경우 정수 임베딩이 0임)
attention_masks = []

# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션 연산을 수행하지 않아 연산 면에서 이득임.
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# 훈련 데이터 세트를 훈련, 검증 데이터 세트로 분할함
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

# 데이터를 텐서화
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)	

batch_size = 64

# 파이토치의 DataLoader로 배치 사이즈 만큼 데이터를 load
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 테스트 데이터에 대해서 같은 과정을 수행
sentences = X_test['product_name']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
labels = y_test.values
tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
MAX_LEN = 64
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)
batch_size = 64
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# GPU 사용 가능 여부 확인
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    logging.info('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
if torch.cuda.is_available():    
    device = torch.device("cuda")
    logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
    logging.info('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    logging.info('No GPU available, using the CPU instead.')

# 가장 최근의 모델을 가져와 추가 훈련
model = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base', num_labels=318)
model.load_state_dict(torch.load('kor_model_latest.pth'))
model.cuda()

''' 이 부분은 만약 카테고리의 수가 늘어났을 경우 최근의 모델을 로드하여 분류기 부분만 훈련을 하기 위한 코드임.

model = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base', num_labels=317)
model.load_state_dict(torch.load('kor_model_latest.pth'))
model.classifier = torch.nn.Linear(768, 318, bias=True)
model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
model.classifier.bias.data.zero_()
# 변경된 모델 저장
model.save_pretrained('new_bert_model')
# 변경된 모델 다시 로드
model = BertForSequenceClassification.from_pretrained('new_bert_model', num_labels=318)
model.cuda()
'''

optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 에포크 수 (pretrained 된 모델이기 때문에 많은 에포크 수를 진행하지 않음)
epochs = 15
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
     
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
criterion = torch.nn.CrossEntropyLoss()

# Training Code
model.zero_grad()
for epoch_i in range(0, epochs):
    logging.info("")
    logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    logging.info('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0
    total_accuracy = 0

    # 훈련모드로 변경
    model.train()
        
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # Forward 수행                
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        # 로스 구함
        loss = outputs[0]
        logits = outputs.logits

        # 총 로스 계산
        total_loss += loss.item()
        # 로짓을 CPU로 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 배치 정확도 계산
        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        total_accuracy += tmp_train_accuracy

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)   
    avg_train_accuracy = total_accuracy / len(train_dataloader)         

    logging.info("")
    logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
    logging.info("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
    logging.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # Validation Code
    logging.info("")
    logging.info("Running Validation...")

    #시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
    # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0]
        # 로스 구함
        tmp_eval_loss = criterion(logits.view(-1, 318), b_labels.view(-1))
        eval_loss += tmp_eval_loss

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    logging.info("  Loss: {0:.2f}".format(eval_loss / nb_eval_steps))
    logging.info("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    logging.info("  Validation took: {:}".format(format_time(time.time() - t0)))

logging.info("")
logging.info("Training complete!")

t0 = time.time()
model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
failed_predictions = []

for step, batch in enumerate(test_dataloader):
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    batch = tuple(t.to(device) for t in batch)
    
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():  
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs[0]
    tmp_eval_loss = criterion(logits.view(-1, 318), b_labels.view(-1))
    eval_loss += tmp_eval_loss

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

    predictions = np.argmax(logits, axis=1)
    
    for i in range(len(label_ids)):
        if predictions[i] != label_ids[i]:
            failed_predictions.append({
                'input_ids': b_input_ids[i].cpu().numpy(),
                'predicted_label': predictions[i],
                'true_label': label_ids[i]
            })

logging.info("")
logging.info("Loss: {0:.2f}".format(eval_loss / nb_eval_steps))
logging.info("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
logging.info("Test took: {:}".format(format_time(time.time() - t0)))


# 라벨 데이터 로드
data = pd.read_csv("label.csv")

# 예측에 실패한 샘플 출력
logging.info("\nFailed Predictions:")
failed_predictions_log = []
for idx, failed in enumerate(failed_predictions):
    input_text = tokenizer.decode(failed['input_ids'], skip_special_tokens=True)
    predicted_label_name = data[data['number'] == failed['predicted_label']]['label'].values[0]
    true_label_name = data[data['number'] == failed['true_label']]['label'].values[0]
    failed_predictions_log.append({
                    'input_ids': tokenizer.decode(failed['input_ids'], skip_special_tokens=True),
                    'predicted_label': data[data['number'] == failed['predicted_label']]['label'].values[0],
                    'true_label': data[data['number'] == failed['true_label']]['label'].values[0]
                })

# 예측에 실패한 샘플을 데이터프레임으로 변환
failed_df = pd.DataFrame(failed_predictions_log)

# 데이터프레임을 CSV 파일로 저장
failed_df.to_csv('./result/failed_predictions.csv', index=False)

logging.info("\nFailed predictions have been saved to 'failed_predictions.csv'")

model_name = 'model.pth'

torch.save(model.state_dict(), model_name)

# GCP 스토리지에 접근하기 위해 필요한 json 파일
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./************************.json"

bucket_name = 'GCP 버킷 이름'    # 서비스 계정 생성한 bucket 이름 입력
source_file_name = f'./{model_name}'    # GCP에 업로드할 파일 절대경로
destination_blob_name = 'model.pth'    # 업로드할 파일을 GCP에 저장할 때의 이름

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)

blob.upload_from_filename(source_file_name)