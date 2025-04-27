from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from GitHub Pages

# Simple Q-learning setup
Q_TABLE = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 0.1
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01
ACTIONS = 6

def state_to_key(state):
    player_pos = tuple(round(x, 1) for x in state['playerPos'])
    point_pos = tuple(round(x, 1) for x in state['closestPointPos'])
    obstacle_pos = tuple(round(x, 1) for x in state['closestObstaclePos'])
    return str((player_pos, point_pos, obstacle_pos))

@app.route('/predict', methods=['POST'])
def predict():
    global EXPLORATION_RATE
    data = request.get_json()
    state = data['state']
    state_key = state_to_key(state)

    if state_key not in Q_TABLE:
        Q_TABLE[state_key] = np.zeros(ACTIONS)

    q_values = Q_TABLE[state_key]

    if np.random.rand() < EXPLORATION_RATE:
        action = np.random.randint(ACTIONS)
    else:
        action = np.argmax(q_values)

    EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)

    return jsonify({
        'action': int(action),
        'qValues': q_values.tolist(),
        'explorationRate': EXPLORATION_RATE
    })

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    state = data['state']
    action = data['action']
    reward = data['reward']
    next_state = data['nextState']

    state_key = state_to_key(state)
    next_state_key = state_to_key(next_state)

    if state_key not in Q_TABLE:
        Q_TABLE[state_key] = np.zeros(ACTIONS)
    if next_state_key not in Q_TABLE:
        Q_TABLE[next_state_key] = np.zeros(ACTIONS)

    current_q = Q_TABLE[state_key][action]
    max_next_q = np.max(Q_TABLE[next_state_key])
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
    Q_TABLE[state_key][action] = new_q

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
