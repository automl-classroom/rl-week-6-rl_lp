import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daten laden
df = pd.read_csv("actor_critic_comparison.csv")

baselines = df["baseline"].unique()
steps = sorted(df["step"].unique())

plt.figure(figsize=(10, 6))

# Stil und Farben für verschiedene Baselines
styles = {"none": "-", "avg": "--", "value": ":", "gae": "-."}
colors = {"none": "blue", "avg": "orange", "value": "green", "gae": "red"}

# Für jeden Baseline-Typ plotten
for b in baselines:
    # Daten pivotieren (wie in deinem Code)
    pivot = df[df["baseline"] == b].pivot(
        index="seed", columns="step", values="mean_return"
    )

    # Statistiken berechnen
    means = pivot.mean().values
    std_errs = pivot.std().values / np.sqrt(pivot.shape[0])

    # Mit Konfidenzintervall plotten
    plt.plot(steps, means, styles.get(b, "-"), label=b, color=colors.get(b))
    plt.fill_between(
        steps, means - std_errs, means + std_errs, alpha=0.2, color=colors.get(b)
    )

# Beschriftung und Layout
plt.title("Actor-Critic Baseline Comparison (LunarLander-v3)")
plt.xlabel("Training Steps")
plt.ylabel("Average Return")
plt.legend()
plt.grid(alpha=0.3)

# Speichern und Anzeigen
save_path = "actor_critic_baseline_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
