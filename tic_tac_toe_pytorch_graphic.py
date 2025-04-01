import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
import os

# Hyperparamètres
alpha = 0.001  # Taux d'apprentissage
gamma = 0.9   # Facteur de réduction
epsilon = 0.1 # Taux d'exploration
_episodes = 10000  # Nombre d'épisodes d'entraînement
MODEL_PATH = 'tic_tac_toe_q_learning.pth'

##### Partie Q-learning #####

# Réseau de neurones pour approximer les Q-valeurs
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128) # Entrée de 9 (plateau 3x3)
        self.fc2 = nn.Linear(128, 64) # Couche cachée
        self.fc3 = nn.Linear(64, 9) # Sortie de 9 (actions possibles)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Activation ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # Pas d'activation à la sortie
        return x

# Choisir une action (exploration/exploitation)
def predict_action(state, model):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 8)  # Position aléatoire
    else:
        with torch.no_grad(): # Désactive le calcul des gradients (car on fe fait que prédire, pas entraîner)
            q_values = model(torch.tensor(state, dtype=torch.float32)) # Applique forward sur le modèle
            return torch.argmax(q_values).item() # Renvoie l'index de la meilleure action, .item pour convertir en entier

# Met à jour le plateau après une action
def take_action(state, action, player):
    new_state = state[:] # Copie de l'état actuel
    new_state[action] = player # Met à jour la case choisie selon le joueur
    return new_state

# Vérifie si le jeu est terminé
# Regarde dans chaque ligne, colonne ou diagonale si trois cases sont identiques
def check_winner(state):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for pos in win_positions:
        if state[pos[0]] == state[pos[1]] == state[pos[2]] != 0:
            return state[pos[0]] # Renvoie le joueur gagnant
    return 0 # Pas de gagnant
# Entraînement du modèle avec mise à jour de l'UI
def train_model(episodes=_episodes, update_progress=None):
    model = TicTacToeNet()
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss() # Mean Squared Error Loss. Fonction utilisée pour calculer la perte (différence entre la prédiction et la cible). Pénalise les erreurs importantes car au carré

    for episode in range(episodes):
        state = [0] * 9  # Plateau vide
        player = 1

        while True:
            action = predict_action(state, model) # Soit aléatoire (si pas assez d'exploration), soit basée sur le modèle
            next_state = take_action(state, action, player)
            reward = 0

            winner = check_winner(next_state)
            if winner == player:
                reward = 1  # Victoire
            elif winner != 0:
                reward = -1 # Défaite
            elif 0 not in next_state:
                reward = 0  # Match nul

            # Calcul de la cible
            with torch.no_grad():
                q_next = model(torch.tensor(next_state, dtype=torch.float32)) # Prédiction des Q-valeurs pour l'état suivant
                max_q_next = torch.max(q_next).item() # Meilleure action pour l'état suivant
                target = reward + gamma * max_q_next # Q-valeur cible

            # Propagation avant et calcul de la perte
            q_values = model(torch.tensor(state, dtype=torch.float32))
            loss = loss_fn(q_values[action], torch.tensor(target))

            # Rétropropagation
            optimizer.zero_grad() # Réinitialise les gradients
            loss.backward() # Calcul des gradients des paramètres du modèle par rapport à la perte
            optimizer.step() # Met à jour les poids du modèle en fonction des gradients

            state = next_state[:]
            player = -player  # Changer de joueur

            if reward != 0 or 0 not in state: # Si le match a été gagné ou le plateau est plein
                break

        if update_progress:
            update_progress(episode / _episodes * 100)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Entraînement terminé!")

model = TicTacToeNet()


##### Partie Interface Graphique #####

# Recharge le modèle depuis le fichier
def reload_model():
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        training_label.config(text="Modèle rechargé depuis le fichier.")
        print("Modèle rechargé depuis le fichier.")
    else:
        training_label.config(text="Aucun fichier de modèle trouvé.")
        print("Aucun fichier de modèle trouvé.")

# Réinitialise le jeu
def reset_game():
    global state, player
    state = [0] * 9

    # Détermine qui commence en fonction du choix dans le combobox
    if starting_player.get() == "Je commence":
        player = 1  # Le joueur commence
    else:
        player = -1  # Le modèle commence
        root.after(100, ai_play)  # Appelle ai_play() avec un léger délai

    for btn in buttons:
        btn.config(text="", state="normal")
    training_label.config(text="En attente...")

# Gestion des clics sur les boutons et appel du modèle
def on_click(index):
    global player
    if state[index] == 0 and player == 1:  # Vérifie que c'est le tour du joueur
        # Coup du joueur
        state[index] = player
        buttons[index].config(text="X")
        check_game_over()

        # Passer au tour du modèle si le jeu n'est pas terminé
        if 0 in state and check_winner(state) == 0:
            player = -1  # Tour du modèle
            ai_play()

# Fait jouer le modèle
def ai_play():
    print("Le modèle joue...")
    global player
    action = predict_action(state, model)  # Le modèle choisit une action
    while state[action] != 0:  # S'assure que l'action est valide
        action = predict_action(state, model)
    print(f"L'IA a choisi l'action {action}")
    # Coup de l'IA
    state[action] = player
    print(f"État après le coup de l'IA : {state}")
    buttons[action].config(text="O")
    check_game_over()

    # Retour au joueur
    if 0 in state and check_winner(state) == 0:
        player = 1

# Progress bar pour l'entraînement
def update_progress(value):
    progress_bar['value'] = value
    root.update_idletasks()

# Fonction pour démarrer l'entraînement
def start_training():
    training_label.config(text="")  # Réinitialise le texte avant de commencer
    training_label.config(text="Entraînement en cours...")
    train_model(episodes=_episodes, update_progress=update_progress)
    training_label.config(text="Entraînement terminé!")

root = tk.Tk()
root.title("Tic Tac Toe avec Q-learning")

# Variable pour stocker le choix du joueur qui commence
starting_player = tk.StringVar(value="Je commence")  # Valeur par défaut

# Conteneur pour le bouton d'entraînement et la barre de progression
top_frame = ttk.Frame(root, padding=10)
top_frame.pack(side="top", fill="x", pady=10)

train_button = ttk.Button(top_frame, text="Lancer l'entraînement", command=start_training)
train_button.pack(pady=5)

reload_button = ttk.Button(top_frame, text="Recharger le modèle depuis le fichier", command=reload_model)
reload_button.pack(pady=5)

reset_button = ttk.Button(top_frame, text="Rejouer", command=reset_game)
reset_button.pack(pady=5)

# Sélecteur pour choisir le joueur qui commence
starting_player_label = ttk.Label(top_frame, text="Qui commence ?")
starting_player_label.pack(pady=5)

starting_player_selector = ttk.Combobox(top_frame, textvariable=starting_player, state="readonly")
starting_player_selector['values'] = ("Je commence", "Le modèle commence")
starting_player_selector.pack(pady=5)

training_label = ttk.Label(top_frame, text="En attente...")
training_label.pack()

progress_bar = ttk.Progressbar(top_frame, orient='horizontal', length=200, mode='determinate')
progress_bar.pack(pady=5)

# Conteneur pour la grille de jeu
game_frame = ttk.Frame(root, padding=10)
game_frame.pack(side="top", pady=10)

# Grille de jeu 3x3 avec des boutons
buttons = []
state = [0] * 9
player = 1

for i in range(9):
    btn = tk.Button(game_frame, text="", width=10, height=4, command=lambda i=i: on_click(i))
    btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
    buttons.append(btn)

# Vérifie l'état de la partie et affiche le gagnant
def check_game_over():
    winner = check_winner(state)
    if winner != 0:
        training_label.config(text=f"Joueur {'X' if winner == 1 else 'O'} a gagné!")
        disable_buttons()
    elif 0 not in state:
        training_label.config(text="Match nul!")
        disable_buttons()

# Désactive tous les boutons après la fin de la partie
def disable_buttons():
    for btn in buttons:
        btn.config(state='disabled')

root.mainloop()