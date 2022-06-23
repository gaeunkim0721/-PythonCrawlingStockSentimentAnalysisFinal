import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gluonnlp as nlp
import numpy as np
from torch import nn
import nee, nee_ksm
import stockRank
from kobert import get_pytorch_kobert_model
import csv
from kobert import get_tokenizer

device = torch.device('cpu')


bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5



class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)        

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# 위에서 설정한 tok, max_len, batch_size, device를 그대로 입력
# comment : 예측하고자 하는 텍스트 데이터 리스트
def getSentimentValue(comment, tok, max_len, batch_size, device):
  commnetslist = [] # 텍스트 데이터를 담을 리스트
  emo_list = [] # 감성 값을 담을 리스트
  # for c in comment: # 모든 댓글
  commnetslist.append( [comment, 5] ) # [댓글, 임의의 양의 정수값] 설정
    
  pdData = pd.DataFrame( commnetslist, columns = [['댓글', '감성']] )
  pdData = pdData.values
  test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False) 
  test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)
  
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length 
    # 이때, out이 예측 결과 리스트
    out = model(token_ids, valid_length, segment_ids)
	
    # e는 2가지 실수 값으로 구성된 리스트
    # 0번 인덱스가 더 크면 부정, 긍정은 반대
    for e in out:
      if e[0]>e[1]: # 부정
        value = 0
      else: #긍정
        value = 1
      emo_list.append(value)

  return emo_list # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환



gaeunPath = r'C:\Users\Admin\Desktop\team3\chromedriver.exe' # 가은님 파일위치 복붙해왔어요 혹시 안되면 다시 확인해보세요 

# 거래량 상위 20위 종목
stocksList = stockRank.getStocks(gaeunPath)
# 종목 기사 헤드라인 크롤링
filename = nee.start(stocksList)


file = open(filename, encoding='utf-8')

type(file)

csvreader = csv.reader(file)

rows = []
names = []
for row in csvreader:
  if row[0] != '종목이름' and row[1]!= '기사제목':
    names.append(row[0])
    rows.append(row[1])


namelist = []
for v in names:
    if v not in namelist:
        namelist.append(v)

print(namelist)

for s in rows:
  print(getSentimentValue(s, tok, max_len, batch_size, device),s)

counts=[0 for i in range(20)]
newscounts=[0 for i in range(20)]

for c, n in enumerate(namelist):    
    for i, s in enumerate(rows):
        if n == names[i]:
            newscounts[c] = newscounts[c] + 1
            if getSentimentValue(s, tok, max_len, batch_size, device)[0] == 1:
                counts[c] = counts[c] + 1

print(counts, newscounts)


percentAverage = 0

for i in range(len(counts)):
    if newscounts[i]!=0:
        percentAverage += counts[i]/newscounts[i]

percentAverage /= 20


recommendations = []
percentage = []
for i in range(len(counts)):
    if newscounts[i]!=0:
        if (counts[i]/newscounts[i] > percentAverage and newscounts[i] > 50):
            if (len(recommendations) < 6 and len(percentage) < 6):
                recommendations.append(namelist[i])
                percentage.append(int(counts[i]/newscounts[i]*100))


print(recommendations, percentage)
print(namelist)

dictionary0 = dict(zip(percentage, recommendations))
dictionary0 = sorted(dictionary0.items(), reverse=True)

print(dictionary0)


#카카오 메세지 전송
nee_ksm.sendmsg(nee_ksm.token_refresh(),dictionary0)