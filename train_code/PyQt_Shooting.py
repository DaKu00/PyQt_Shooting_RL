# 포물선 운동 애니메이션
# 이동 애니메이션
import time
import sys
import time
import math
import gym
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Target:
    def __init__(self):
        self.x = 0
        self.y = 300  # 게임윈도우의 크기를 400 X 300으로 했기때문에 게임윈도우의 바닥을 나타냄

    #         self.inc = 1
    def move(self):
        self.x += 1  # self.inc
        self.y = int((1 / 132) * (self.x - 200) ** 2 + 0)

    # 총알이 발사 되었을 때의 타겟 위치를 리턴하는 함수
    def get_state(self):
        return self.x, self.y

    def reset(self):
        self.x = 0
        self.y = 300

class Bullet:
    def __init__(self, rad):
        self.rad = rad
        self.x = 195
        self.y = 290
        self.vx = math.cos(rad) * 3
        self.vy = -math.sin(rad) * 3
        self.target_state = (0, 0)

    # 총알의 x, y의 변화량으로 움직임
    def move(self):
        self.x += self.vx
        self.y += self.vy


# envifonment 역할
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.target = Target()
        self.bullets = []

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(QPixmap(400, 300))
        # hit했는지 확인하는 변수
        self.ishit = False

        self.episodes = 400
        self.fireCnt = 0
        self.hits = 0

        # 탄환을 발사할 당시의 타겟의 좌표를 저장할 리스트
        self.hit_state = []

        # 타겟을 명중 하였을 때의 발사 각도를 저장할 리스트
        self.hit_rad = []

        self.start_check = False

        self.move_action = QAction('&Start', self)
        # move_action.setStatusTip("Click to start")
        self.move_action.triggered.connect(self.action_move)


        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.addAction(self.move_action)

        self.setCentralWidget(self.label)

        self.fire_model = 0
        self.rad_model = 0

    def set_models(self, fire_model, rad_model):
        self.fire_model = fire_model
        self.rad_model = rad_model

    # 클래스를 만들고 있으므로 이벤트 핸들러를 명시하여 사용, 클래스를 안만들고 사용할 경우, 시그널을 통해 연결한다.
    # repaint를 호출함으로서 paintEvent함수 호출
    def paintEvent(self, event):
        painter = QPainter(self.label.pixmap())

        # 화면 지우기
        painter.fillRect(QRect(0, 0, 400, 300), Qt.black)

        # painter.setPen(QPen(Qt.green,  8, Qt.SolidLine))  # 테두리 선
        brush = None

        # 명중하였을 때 돌아가는 코드(ishit은 명중 판별 변수)
        if self.ishit:
            # 폭발시 흰색으로 타겟을 그림, 타겟 전체를 하얗게 칠함
            brush = QBrush(Qt.white, Qt.SolidPattern)
            # ishit을 바로 False로 바꿔줌으로서 명중시 잠깐 하얗게 하고 원래대로 돌아오게 함
            self.ishit = False
        else:
            brush = QBrush(Qt.red, Qt.SolidPattern)

        painter.setBrush(brush)  # 채울 색상, 패턴
        # drawEllipse(QRect), QRect(x, y, w, h)
        painter.drawEllipse(self.target.x, self.target.y, 20, 20)

        # 타겟보다 약간 작은 크기로 총알을 그림
        for b in self.bullets:
            painter.setPen(Qt.white)
            painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
            painter.drawEllipse(int(b.x), int(b.y), 10, 10)
        painter.end()

    def action_move(self):
        self.start_check = True
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.test)
        self.timer.start()

    def move(self):
        self.target.move()
        for b in self.bullets:
            b.move()
            # 탄환이 윈도우 위쪽을 넘어가면, 오른쪽, 왼쪽 창을 넘어갈 경우 총알을 제거하여 관리
            if b.y <= -10 or b.x <= 0 or b.x >= 400:
                self.bullets.remove(b)
            # 타겟과 발사체의 충돌(명중) 검사
            elif ((self.target.x - b.x) ** 2 + (self.target.y - b.y) ** 2) < (20 ** 2):
                # 명중하였을 때의 타겟의 위치와 탄환 발사시의 각도를 저장
                self.hit_state.append(b.target_state)
                self.hit_rad.append(b.rad)
                self.bullets.remove(b)

                self.ishit = True
                self.hits += 1
        # 화면을 다시 그리라고 요청하며 paintEvent함수를 호출
        self.repaint()

    # 스페이스바를 누르면 발사하는 코드를 담은 키프레스이벤트 함수
    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Space:
            #             rad = math.radians(self.rad)
            while True:
                state = self.target.get_state()
                prob = self.fire_model.predict(np.array(state).reshape(-1, 2))
                if np.argmax(prob[0]) == 1:
                    rad = self.rad_model.predict(np.array(state).reshape(-1, 2))
                    self.step(1, rad)

                    break
    #                 elif: break

    def fire(self, rad):
        # 스페이스바를 누르면 발사됨, 발사될 때의 각을 탄환객체에 저자
        b = Bullet(rad)
        # 탄혼 발사 당시의 타겟의 위치 정보를 저장
        b.target_state = self.target.get_state()
        self.bullets.append(b)
        self.fireCnt += 1

    def reset(self):
        self.target.reset()
        self.fireCnt = 0
        self.hits = 0
        self.bullets.clear()
        self.episodes = 400
        return self.target.get_state()

    def step(self, action, rad):
        self.episodes -= 1
        done = False
        if self.episodes == 0:
            done = True
        if action == 1:
            self.fire(rad)
        self.move()
        return self.target.get_state(), 0, done, {}

# Thread를 이용한 Window 실행
# QMainWindow를 실행할 때는 QApplication 인스턴스의 exec_()함수를 실행해야 한다
# exec_()함수는 이벤트 루프(무한루프)를 실행하므로 exec_() 아래에 있는 코드는 실행되지 않는다
# 그러므로 QMainWindow 와 함께 다른 코드를 실행하려면 아래와 같이 한개의 Thread로 실행하면 된다

import threading

window = None

def start_window():
    app = QApplication(sys.argv)
    global window
    window = MainWindow()
#     window.set_models(fire_model, rad_model)
    window.show()
    app.exec_()

win_thread = threading.Thread(target=start_window)
win_thread.start()


# 신경망 학습을 위해 게임으로부터 데이터 수집
# window.hit_state : 명중한 발사체로부터 수집된 타겟 상태(x, y)
# window.hit_rad : 명중한 발사체로부터 수집된 발사 각도(radian)

import time

print('무작위 대응으로 게임 데이터 수집')
states = []
# 데이터가 많을수록 학습효과와 정확도가 높다
for e in range(300):
    state = window.reset()
    done = False
    while not done:
        rad = math.radians(np.random.randint(0,180))  # 무작위 발사 각
        action = 1
        state,reward,done,info = window.step(1,rad)   # action=1 으로 발사 요청
        states.append(state)
        #time.sleep(0.02)
    print('Episode %s fires %s hits %s' % (e+1, window.fireCnt, window.hits) )
print('데이터 수집 종료')

#print(window.hit_state)
#print(window.hit_rad)



# 위에서 모은 데이터의 라벨링

import numpy as np

fire_label = []

for s in states:
    ohe = np.zeros(2)
    if s in window.hit_state:
        ohe[1] = 1
    else:
        ohe[0] = 1
    fire_label.append(ohe)

state_x = np.array(states)
fire_y = np.array(fire_label)

print(type(state_x), state_x.shape)
print(type(fire_y), fire_y.shape)


# 학습 데이터들의 Type을 확인
# 발사각 계산용 신경망을 학습시킬 데이터 준비( 문제(x,y), 정답(rad))
hit_state_x = np.array(window.hit_state)
# hit_rad는 1차원이므로, reshape을 통해 2차원 배열로 만들어줌
hit_rad_y = np.array(window.hit_rad).reshape(-1,1)

print(type(hit_state_x), hit_state_x.shape)
print(type(hit_rad_y), hit_rad_y.shape)


# 발사 여부를 계산할 신경망 생성(Action)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# 뉴런의 수와 은닉층의 수는 테스트를 진행하면서 변경해가면서 적정수를 찾는다
def create_fire_model():
    fire_model = Sequential()
    fire_model.add(Dense(64, activation='relu', input_shape=(2,)))
    fire_model.add(Dense(64, activation='relu'))
    fire_model.add(Dense(64, activation='relu'))
    fire_model.add(Dense(64, activation='relu'))
    fire_model.add(Dense(2, activation="softmax"))

    fire_model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return fire_model

# 발사여부를 계산할 신경망 학습
fire_model = create_fire_model()
fire_model.fit(state_x, fire_y, epochs=100)


# 발사 각도(rad)를 계산할 신경망 생성

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def create_rad_model():
    rad_model = Sequential()
    rad_model.add(Dense(64, activation='relu', input_shape=(2,)))
    rad_model.add(Dense(64, activation='relu'))
    rad_model.add(Dense(64, activation='relu'))
    rad_model.add(Dense(64, activation='relu'))
    rad_model.add(Dense(1, activation="linear"))

    rad_model.compile(
        loss="mse",
        optimizer="adam",
        metrics=["mae"])
    return rad_model

# 발사각을 계산할 신경망 학습
rad_model = create_rad_model()
rad_model.fit(hit_stateX, hit_radY, epochs=200)


# 학습된 모델의 가중치(Weight)를 파일에 저장
fire_model.save('fire_model_weights.h5')
rad_model.save('rad_model_weights.h5')

