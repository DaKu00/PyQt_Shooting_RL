# PyQt_Shooting_RL(PyQt를 활용한 슈팅 강화학습, 시각화, 학습모델 응용)

### 구조
 - 포물선으로 움직이는 타겟을 맞추는 형식
 - 타겟을 향해 탄환을 발사하면 날아가는데까지 이동거리 존재(탄환이 날아가는 동안 타겟도 이동)
 - 무작위로 탄환을 발사하여 탄환이 타겟을 맞추게 되었을 때, 탄환이 발사될 때의 타겟의 위치를 학습에 사용
 - PyQt5를 사용하여 시각화

### 학습
 - 무작위로 탄환을 발사하여 타겟을 맞췄을 때의 state데이터 수집
 - 수집된 데이터를 바탕으로 쏠것인지 안쏠것인지 판별하는 강화학습 모델 학습
 - 아웃풋레이어의 활성함수를 softmax로 하여 쏠지 말지 확률적으로 판단
 - 수집된 데이터를 바탕으로 어떤 각도로 쏠것인지를 판별하는 강화학습 모델 학습
 - 아웃풋레이어의 활성함수르 linear로 하여 각도를 선형적으로 예측
 - 학습이 완료된 두개의 모델을 통해 어느방향을 어느 상황에서 쏠지 스스로 판별이 가능한 강화학습 모델 구성
 - 테스트 코드르 구성하고 PyQt window를 통해 시각화하여 테스트 진행
<img src="https://user-images.githubusercontent.com/87750521/127049253-934f7c7a-a0f1-4a89-99e9-9e5349df713d.png" width="400" height="300">



### 응용
 - 학습이 완료된 모델을 통해 자동으로 타겟을 리드하여 탄환을 발사하는 모드 제작 (Mode A)
 - 학습이 완료된 모델을 통해 자동으로 타겟을 리드하고 사용자가 발사결정을 하여 발사하는 모드 제작 (Mode B)
 - 학습이 완료된 모델과 별개로 사용자가 발사할 탄환의 각도 발사타이밍을 결정해 발사하는 모드 제작 (Mode C)
 - PyQt5를 활용하여 모드를 바꿀 수 있게 하며 테스트를 시각적으로 진행할 수 있도록 제작
<img src="https://user-images.githubusercontent.com/87750521/127086479-a4a0f3e8-43d9-473c-bb60-ad0adce41dd9.png" width="1000" height="300">
