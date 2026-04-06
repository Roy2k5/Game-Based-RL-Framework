# Huong dan su dung moi truong Flappy Bird RL

Tai lieu nay huong dan su dung moi truong Flappy Bird trong thu muc env/flappy_bird de huan luyen agent RL.

## 1. Cau truc file

- flappy_bird.py: Moi truong Flappy Bird, reward shaping, render, va 3 che do chay.
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
python env/flappy_bird/flappy_bird.py
```

Chuong trinh se hoi che do:

- 1: Nguoi choi game
- 2: Huan luyen
- 3: Mo hinh choi game

## 4. API moi truong de train RL

Class chinh: FlappyBirdEnv

### Khoi tao

```python
env = FlappyBirdEnv(
    width=420,
    height=640,
    gravity=0.55,
    flap_velocity=-8.5,
    pipe_speed=3.0,
    pipe_gap=170,
)
```

### reset()

```python
state = env.reset()
```

- Tra ve state dau tien dang tuple 4 chieu.

### step(action)

```python
next_state, reward, done, info = env.step(action)
```

- action trong {0, 1}
  - 0: khong flap
  - 1: flap
- next_state: tuple state moi
- reward: diem thuong/phat
- done: True neu va cham
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

## 5. Dinh dang state (4 features)

State gom 4 gia tri rời rạc:

1. dx_to_pipe: khoang cach ngang den ong tiep theo (da chia bin)
2. dy_to_gap_center: chen lech cao do bird voi tam khe ong (da chia bin)
3. velocity_bin: van toc chim (da chia bin)
4. is_pipe_ahead: 1 neu ong dang o phia truoc chim, nguoc lai 0

State dang tuple int nen phu hop truc tiep voi Q-table dictionary key.

## 6. Reward shaping

Class: CustomReward

Mac dinh:

- survive = +0.05
- pass_pipe = +10.0
- crash = -10.0
- center_bonus = +0.1

Tong quan:

- Va cham: reward crash
- Qua ong: reward pass_pipe
- Khac: survive + bonus nho khi bird o gan tam khe

## 7. Train voi Q-table

Da co file train san:

```bash
python train/train_flappy_q_table.py
```

Checkpoint duoc luu tai:

- checkpoint/flappy_q_table.pkl

## 8. Luu y

- Luon goi env.close() khi xong train/test.
- Neu gap loi pygame, cai lai bang pip install pygame.
- Co the giam fps trong train neu may yeu.
