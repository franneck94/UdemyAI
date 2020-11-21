import matplotlib.pyplot as plt
import numpy as np


def plotting_fn(s, ax):
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


def save_map(values, name="test.png"):
    fig, ax = plt.subplots(figsize=(10, 10))
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

    for s in range(len(values)):
        for a in range(len(values[s])):
            posx = s // 4
            posy = s % 4
            max_index = np.argmax(values[s])
            if a == 0: # Left
                weight = "bold" if a == max_index else "normal"
                ax.text(posy - 0.2, posx, "L: " +
                        str(round(values[s][a], 3)), weight=weight, ha="right", va="center")
            elif a == 1: # Down
                weight = "bold" if a == max_index else "normal"
                ax.text(posy, posx + 0.2, "D: " +
                        str(round(values[s][a], 3)), weight=weight, ha="center", va="top")
            elif a == 2: # Right
                weight = "bold" if a == max_index else "normal"
                ax.text(posy + 0.2, posx, "R: " +
                        str(round(values[s][a], 3)), weight=weight, ha="left", va="center")
            elif a == 3: # Up
                weight = "bold" if a == max_index else "normal"
                ax.text(posy, posx - 0.2, "U: " +
                        str(round(values[s][a], 3)), weight=weight, ha="center", va="bottom")
    fig.savefig("./" + name)
    plt.close()


def plotting_q_values(state, action, values, ax):
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

    for s in range(len(values)):
        for a in range(len(values[s])):
            posx = s // 4
            posy = s % 4
            max_index = np.argmax(values[s])
            if a == 0: # Left
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(posy - 0.2, posx, "L: " +
                        str(round(values[s][a], 3)), weight=weight, ha="right", va="center",
                        color=color)
            elif a == 1: # Down
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(posy, posx + 0.2, "D: " +
                        str(round(values[s][a], 3)), weight=weight, ha="center", va="top",
                        color=color)
            elif a == 2: # Right
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(posy + 0.2, posx, "R: " +
                        str(round(values[s][a], 3)), weight=weight, ha="left", va="center",
                        color=color)
            elif a == 3: # Up
                weight = "bold" if a == max_index else "normal"
                color = "red" if action == a and state == s else "black"
                ax.text(posy, posx - 0.2, "U: " +
                        str(round(values[s][a], 3)), weight=weight, ha="center", va="bottom",
                        color=color)

    plt.pause(2.0)
