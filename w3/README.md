## Q1) 어떤 task를 선택하셨나요?
> MNLI를 선택했습니다.


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델은 distilBert를 encoder로 사용하고, classifier 부분은 데이터의 레이블 수와 동일하게 출력을 3으로 해주었습니다.


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> 영어에 대한 이해와 제한된 GPU 내에서 활용가능한 distilBERT를 사용했습니다.


## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> loss curve는 scratch(pre-train 하지 않은 Transformer) 모델이 더 빠르게 감소하는 것으로 나타났습니다.
> 하지만, test accuracy를 비교한 결과 fine-tuning한 모델은 47.7%, scratch 모델은 39%로 일반적인 학습이 잘 이루어졌다기보다는
> 학습데이터에 치중한 학습이 이루어졌다고 볼 수 있습니다.  

