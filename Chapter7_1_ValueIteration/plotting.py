from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def action_map(action: int) -> str:
    if action == 0:
        return "Left"
    elif action == 1:
        return "Down"
    elif action == 2:
        return "Right"
    else:
        return "Up"


def plotting_fn(s: Any, ax: Any) -> None:
    mat = np.full((4, 4), 1)
    mat[1][3] = 0
    mat[2][3] = 0
    mat[3][0] = 0
    mat[1][1] = 0
    mat[3][3] = 2
    posx = s // 4
    posy = s % 4
    mat[posx][posy] = 3
    ax.cla()
    ax.imshow(mat, cmap="Set1")
    ax.text(posy, posx, "Agent", ha="center", va="center")
    ax.text(3, 1, "Hole", ha="center", va="center")
    ax.text(3, 2, "Hole", ha="center", va="center")
    ax.text(0, 3, "Hole", ha="center", va="center")
    ax.text(1, 1, "Hole", ha="center", va="center")
    ax.text(3, 3, "Goal", ha="center", va="center")
    plt.pause(0.3)


def save_map(values: Any, name: str = "test.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    mat = np.full((4, 4), 1)
    mat[1][3] = 0
    mat[2][3] = 0
    mat[3][0] = 0
    mat[1][1] = 0
    mat[3][3] = 2
    ax.cla()
    ax.imshow(mat, cmap="Set1")
    ax.text(3, 1, "Hole", ha="center", va="center")
    ax.text(3, 2, "Hole", ha="center", va="center")
    ax.text(0, 3, "Hole", ha="center", va="center")
    ax.text(1, 1, "Hole", ha="center", va="center")
    ax.text(3, 3, "Goal", ha="center", va="center")

    for s in values.keys():
        for a in values[s].keys():
            posx = s // 4
            posy = s % 4
            max_index = np.argmax(list(values[s].values()))
            if a == 0:  # Left
                weight = "bold" if a == max_index else "normal"
                ax.text(
                    posy - 0.2,
                    posx,
                    "L: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="right",
                    va="center",
                )
            elif a == 1:  # Down
                weight = "bold" if a == max_index else "normal"
                ax.text(
                    posy,
                    posx + 0.2,
                    "D: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="center",
                    va="top",
                )
            elif a == 2:  # Right
                weight = "bold" if a == max_index else "normal"
                ax.text(
                    posy + 0.2,
                    posx,
                    "R: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="left",
                    va="center",
                )
            elif a == 3:  # Up
                weight = "bold" if a == max_index else "normal"
                ax.text(
                    posy,
                    posx - 0.2,
                    "U: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="center",
                    va="bottom",
                )
    fig.savefig("./" + name)


def plotting_q_values(state: Any, action: Any, values: Any, ax: Any) -> None:
    mat = np.full((4, 4), 1)
    mat[1][3] = 0
    mat[2][3] = 0
    mat[3][0] = 0
    mat[1][1] = 0
    mat[3][3] = 2
    posx = state // 4
    posy = state % 4
    mat[posx][posy] = 3
    ax.cla()
    ax.imshow(mat, cmap="Set1")
    ax.text(posy, posx, "Agent", ha="center", va="center")
    ax.text(3, 1, "Hole", ha="center", va="center")
    ax.text(3, 2, "Hole", ha="center", va="center")
    ax.text(0, 3, "Hole", ha="center", va="center")
    ax.text(1, 1, "Hole", ha="center", va="center")
    ax.text(3, 3, "Goal", ha="center", va="center")

    for s in values.keys():
        for a in values[s].keys():
            posx = s // 4
            posy = s % 4
            max_index = np.argmax(list(values[s].values()))
            if a == 0:  # Left
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(
                    posy - 0.2,
                    posx,
                    "L: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="right",
                    va="center",
                    color=color,
                )
            elif a == 1:  # Down
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(
                    posy,
                    posx + 0.2,
                    "D: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="center",
                    va="top",
                    color=color,
                )
            elif a == 2:  # Right
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(
                    posy + 0.2,
                    posx,
                    "R: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="left",
                    va="center",
                    color=color,
                )
            elif a == 3:  # Up
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(
                    posy,
                    posx - 0.2,
                    "U: " + str(round(values[s][a], 3)),
                    weight=weight,
                    ha="center",
                    va="bottom",
                    color=color,
                )
    plt.pause(2.0)
