# Thanks, OpenAI
# Backbone code based on OpenAI Code
import numpy as np
import torch
from torch.optim import Adam
from src.ppo import core
import os


class PPOBuffer:
    """
    Buffer for training Agent
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, discrete=False):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            core.combined_shape(size, None if discrete else act_dim),
            dtype=np.int64 if discrete else np.float32,
        )
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.discrete = discrete
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = int(act) if self.discrete else act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory to calculate Advantage and Return
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # Advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        # Reward
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Normalize Advantage
        adv_mean, adv_std = np.mean(self.adv_buf, keepdims=True), np.std(
            self.adv_buf, keepdims=True
        )
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        tensors = {
            "obs": torch.as_tensor(data["obs"], dtype=torch.float32),
            "act": torch.as_tensor(
                data["act"], dtype=torch.long if self.discrete else torch.float32
            ),
            "ret": torch.as_tensor(data["ret"], dtype=torch.float32),
            "adv": torch.as_tensor(data["adv"], dtype=torch.float32),
            "logp": torch.as_tensor(data["logp"], dtype=torch.float32),
        }
        return tensors


# Hàm train ppo trên môi trường Env
def ppo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=5,
    ckpt_dir="checkpoint",
    batch_size=32,
    device=None,
    steps_per_epoch=4000,
    epochs=500,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    save_freq=10,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env = env_fn()
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    # Create Model
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs).to(device)
    ac.load_checkpoint(ckpt_dir, False)

    # Model size
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    discrete_action = isinstance(ac.pi, core.MLPCategoricalActor)
    buf = PPOBuffer(
        obs_dim,
        act_dim,
        steps_per_epoch,
        gamma,
        lam,
        discrete=discrete_action,
    )

    def compute_loss_pi(data):
        obs, act, adv, logp_old = (
            data["obs"].to(device),
            data["act"].to(device),
            data["adv"].to(device),
            data["logp"].to(device),
        )

        if discrete_action:
            act = act.long()

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"].to(device), data["ret"].to(device)
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()
        size = data["obs"].shape[0]

        for _ in range(train_pi_iters):
            indices = torch.randperm(size)

            for start in range(0, size, batch_size):
                idx = indices[start : start + batch_size]
                batch = {k: v[idx] for k, v in data.items()}

                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(batch)
                kl = pi_info["kl"]

                if kl > 1.5 * target_kl:
                    print("Early stopping")
                    return

                loss_pi.backward()
                pi_optimizer.step()

        for _ in range(train_v_iters):
            indices = torch.randperm(size)

            for start in range(0, size, batch_size):
                idx = indices[start : start + batch_size]
                batch = {k: v[idx] for k, v in data.items()}

                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(batch)
                loss_v.backward()
                vf_optimizer.step()

    o, ep_ret, ep_len = env.reset(), 0, 0
    path = os.path.join("checkpoint", "ppo")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        f"Epoch {epoch}/{epochs}: Reward: {ep_ret} at {ep_len} steps",
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if not d:
                    _, v, _ = ac.step(
                        torch.as_tensor(o, dtype=torch.float32).to(device)
                    )
                else:
                    v = 0
                buf.finish_path(v)
                o, ep_ret, ep_len = env.reset(), 0, 0
        update()
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            ac.save_checkpoint(path)


if __name__ == "__main__":
    pass
