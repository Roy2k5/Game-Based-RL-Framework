# Huong dan su dung moi truong Snake RL

Tai lieu nay huong dan ban su dung moi truong Snake trong thu muc env/snake de huan luyen agent RL.

## 1. Cau truc file

- snake.py: Moi truong Snake, reward shaping, render, va 3 che do chay.
- docs.md: Tai lieu huong dan (file nay).

## 2. Yeu cau cai dat

Can Python 3.9+ (khuyen nghi 3.10/3.11).

Cai cac thu vien can thiet:

```bash
pip install numpy pygame
```

## 3. Chay nhanh moi truong

Tu thu muc goc du an RL_algorithm, chay:

```bash
python env/snake/snake.py
```

Chuogn trinh se hoi che do:

- 1: Nguoi choi game
- 2: Huan luyen
- 3: Mo hinh choi game

## 4. Giai thich cac che do

### Mode 1 - Nguoi choi game

Ham duoc goi: run_human_mode()

Dieu khien:

- Mui ten trai: re trai
- Mui ten phai: re phai
- Mui ten len: di thang

Neu ran chet, game se in diem va reset van moi.

### Mode 2 - Huan luyen

Ham duoc goi: run_training_mode(episodes=...)

Hien tai day la scaffold de ban gan agent RL vao:

- Moi buoc dang lay action ngau nhien
- Da co tuple kinh nghiem: (state, action, reward, next_state, done)
- Ban thay action random bang output tu agent cua ban

### Mode 3 - Mo hinh choi game

Ham duoc goi: run_model_mode(model=..., episodes=...)

- Neu khong truyen model, he thong dung RandomPolicy de demo.
- Neu co model, model can la callable nhan state va tra ve action trong tap {0, 1, 2}.

## 5. API moi truong de train RL

Class chinh: SnakeEnv

### Khoi tao

```python
env = SnakeEnv(
    width=20,
    height=20,
    block_size=20,
    fps=12,
    max_steps_without_food=200,
    reward_fn=CustomReward(),
    seed=42,
)
```

### reset()

```python
state = env.reset()
```

- Tra ve state dau tien (numpy array float32, kich thuoc 11).

### step(action)

```python
next_state, reward, done, info = env.step(action)
```

- action: 0, 1, hoac 2
  - 0: di thang
  - 1: re phai
  - 2: re trai
- next_state: vector state moi
- reward: diem thuong/phat
- done: True neu ket thuc van
- info: dict gom score va steps

### render(mode)

```python
env.render(mode="human")
frame = env.render(mode="rgb_array")
```

- mode="human": mo cua so pygame de xem game
- mode="rgb_array": tra frame dang numpy array (H, W, 3)

### close()

```python
env.close()
```

Dong pygame va giai phong tai nguyen sau khi xong.

## 6. Dinh dang state (11 features)

State gom 11 gia tri:

1. danger_straight
2. danger_right
3. danger_left
4. dir_up
5. dir_right
6. dir_down
7. dir_left
8. food_up
9. food_right
10. food_down
11. food_left

Tat ca duoc ma hoa 0/1, dtype float32.

## 7. Reward shaping

Class: CustomReward

Mac dinh:

- eat_food = +10.0
- die = -10.0
- step = -0.05
- toward_food = +0.1
- away_from_food = -0.1

Cong thuc tong quat:

- Neu chet: reward = die
- Neu an moi: reward = eat_food
- Nguoc lai: reward = step + shaping

Trong do shaping tuy theo ran di gan hay xa food theo Manhattan distance.

### Tuy bien reward

```python
reward_fn = CustomReward(
    eat_food=20.0,
    die=-15.0,
    step=-0.01,
    toward_food=0.2,
    away_from_food=-0.2,
)
env = SnakeEnv(reward_fn=reward_fn)
```

## 8. Vi du vong lap train toi thieu

```python
import numpy as np
from env.snake.snake import SnakeEnv

env = SnakeEnv()
for episode in range(100):
    state = env.reset()
    done = False

    while not done:
        action = np.random.randint(0, 3)  # thay bang agent.predict(state)
        next_state, reward, done, info = env.step(int(action))

        # train agent o day
        # agent.update(state, action, reward, next_state, done)

        state = next_state

    print(f"Episode {episode}: score={info['score']}")

env.close()
```

## 9. Vi du mode model choi game

```python
from env.snake.snake import run_model_mode


def my_policy(state):
    # state la numpy array 11 chieu
    # Tra ve 0, 1, hoac 2
    return 0


run_model_mode(model=my_policy, episodes=10)
```

## 10. Luu y quan trong

- Luon goi env.close() sau khi train/test xong.
- Neu gap loi khong import duoc pygame, cai lai bang pip install pygame.
- Trong mode 2, fps dang cao de train nhanh hon (it render).
- Gioi han max_steps_without_food giup tranh agent di long vong vo han.

## 11. Huong phat trien tiep

- Them wrapper Gymnasium (de dung truc tiep voi PPO/A2C tu Stable-Baselines3).
- Luu va nap model de dung voi mode 3.
- Logging TensorBoard cho reward va score.
