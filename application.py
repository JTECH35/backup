from flask import Flask, redirect, url_for, render_template, request
import os
import random



secret_key = str(os.urandom(24))

application = Flask(__name__)

application.config['TESTING'] = True
application.config['DEBUG'] = True
application.config['FLASK_ENV'] = 'development'
application.config['SECRET_KEY'] = secret_key
application.config['DEBUG'] = True





# Defining the home page of our site        #views
@application.route("/", methods=['GET', 'POST'])
def first():
    return render_template("home2.html")

@application.route("/home2", methods=['GET', 'POST'])
def home2():
    return render_template("home2.html")

@application.route('/signup')
def signup():
    return render_template('signup.html')


@application.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check login credentials here
        # If login successful, redirect to home
        return redirect(url_for('home2'))
    return render_template("login.html")

@application.route("/about", methods=['GET', 'POST'])
def about():
    return render_template("about.html")

def run_webcam():
    os.system("python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat")



@application.route("/start", methods=['GET', 'POST'])
def index():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Start') == 'Start':
            run_webcam()
            return render_template("home2.html")
    else:
        # pass # unknown
        return render_template("home2.html")


@application.route("/minigame", methods=['GET', 'POST'])
def minigame():
    return render_template("minigame.html")


@application.route("/questions", methods=['GET', 'POST'])
def questions():
    return render_template("questions.html")


# GUESSING GAME

lower_bound = 1
upper_bound = 100

def generate_secret_number():
    # Generate a random secret number within the specified range
    return random.randint(lower_bound, upper_bound)

# Set the initial secret number
secret_number = generate_secret_number()


@application.route("/guessinggame", methods=["GET", "POST"])
def guessinggame():
    global secret_number  # Make sure to use the global variable

    if request.method == "POST":
        # Get the user's guess from the form
        user_guess = int(request.form["guess"])

        # Check if the guess is correct
        if user_guess == secret_number:
            result_message = "Congratulations! You guessed the correct number."
            # Generate a new secret number for the next round
            secret_number = generate_secret_number()
        elif user_guess < secret_number:
            result_message = "Too low! Try a higher number."
        else:
            result_message = "Too high! Try a lower number."

        return render_template("guessinggame.html", result_message=result_message)

    return render_template("guessinggame.html", lower_bound=lower_bound, upper_bound=upper_bound)








# TIC-TAC-TOE


# Define constants for the players
PLAYER_X = 'X'
PLAYER_O = 'O'

# Initialize the game board
board = [''] * 9
user_symbol = PLAYER_X
ai_symbol = PLAYER_O
current_player = user_symbol

def check_winner(current_board, player):
    # Check rows, columns, and diagonals for a winner
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                             (0, 3, 6), (1, 4, 7), (2, 5, 8),
                             (0, 4, 8), (2, 4, 6)]

    for combo in winning_combinations:
        if current_board[combo[0]] == current_board[combo[1]] == current_board[combo[2]] == player:
            return True

    return False

def check_tie(current_board):
    # Check for a tie by checking if the board is full and no winner
    return all(cell != '' for cell in current_board) and not any([check_winner(current_board, player) for player in [PLAYER_X, PLAYER_O]])

def make_ai_move(current_board):
    # Choose a random empty cell
    empty_cells = [i for i in range(9) if current_board[i] == '']
    if empty_cells:
        return random.choice(empty_cells)
    else:
        return None


def reset_game():
    global board
    global current_player
    board = [''] * 9
    current_player = user_symbol

@application.route('/tictacgame')
def tic_tac_toe_game():
    winner = None
    if check_winner(board, user_symbol):
        winner = user_symbol
    elif check_winner(board, ai_symbol):
        winner = ai_symbol
    elif check_tie(board):
        winner = 'Tie'

    return render_template('tic_tac_toe.html', board=board, winner=winner, tie=check_tie(board))

@application.route('/move/<int:position>')
def make_move(position):
    global current_player
    global board

    if board[position] == '' and not check_winner(board, user_symbol) and not check_tie(board):
        board[position] = user_symbol

        if not check_winner(board, user_symbol) and not check_tie(board):
            # Switch player for the next move
            current_player = ai_symbol

            # AI makes a move
            ai_position = make_ai_move(board)
            if ai_position is not None:
                board[ai_position] = ai_symbol

        return redirect(url_for('tic_tac_toe_game'))

    return render_template('tic_tac_toe.html', board=board, winner=None, tie=check_tie(board))

@application.route('/play_again')
def play_again():
    reset_game()
    return redirect(url_for('tic_tac_toe_game'))


if __name__ == "__main__":
    application.run()


