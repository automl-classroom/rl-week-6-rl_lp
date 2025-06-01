import gymnasium as gym
import pandas as pd
from rl_exercises.week_6.actor_critic import ActorCriticAgent

all_results = {"baseline": [], "seed": [], "step": [], "mean_return": []}

for baseline_type in ["none", "avg", "value", "gae"]:
    for seed in [0, 1, 2, 3, 4]:
        env = gym.make("LunarLander-v3")
        agent = ActorCriticAgent(
            env,
            lr_actor=5e-3,
            lr_critic=1e-2,
            gamma=0.99,
            gae_lambda=0.95,
            seed=seed,
            hidden_size=128,
            baseline_type=baseline_type,
            baseline_decay=0.9,
        )

        step_count = 0
        eval_interval = 5000
        records = []
        while step_count < 50000:
            state, _ = env.reset(seed=seed)
            done = False
            traj = []
            while not done and step_count < 50000:
                action, logp = agent.predict_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                traj.append((state, action, reward, next_state, done, logp))
                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r, _ = agent.evaluate(env, num_episodes=5)
                    records.append((step_count, mean_r))

            agent.update_agent(traj)

        for step, r in records:
            all_results["baseline"].append(baseline_type)
            all_results["seed"].append(seed)
            all_results["step"].append(step)
            all_results["mean_return"].append(r)

df = pd.DataFrame(all_results)
df.to_csv("actor_critic_comparison.csv", index=False)
