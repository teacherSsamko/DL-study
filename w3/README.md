## Q1) 어떤 task를 선택하셨나요?
> MNLI를 선택했습니다.


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델의 입력과 출력 형태 또는 shape을 정확하게 기술


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> 영어에 대한 이해와 제한된 GPU 내에서 활용가능한 distilBERT를 사용했습니다.


## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> 비교 metric은 loss curve, accuracy, 또는 test data에 대한 generalization 성능 등을 활용.
> +)이외에도 기계 번역 같은 문제에서 활용하는 BLEU 등의 metric을 마음껏 활용 가능
- 
-  
-  
- 이미지 첨부시 : ![이미지 설명](경로) / 예시: ![poster](./image.png)

### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다(반드시 출력 결과가 남아있어야 합니다!!) 
