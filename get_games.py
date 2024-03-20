import berserk
import pandas as pd

with open("token.txt", "r", encoding="utf-8") as f:
    API_TOKEN = f.readlines()[0][:-1]

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


def color2bool(color):
    if color == "white":
        return True
    else:
        return False


def bool2color(b):
    if b:
        return "white"
    else:
        return "black"


def flip_color(color):
    if color == "white":
        return "black"
    else:
        return "white"


def my_color(playes_data):
    if playes_data["white"]["user"]["name"] == "menisadi":
        return "white"
    else:
        return "black"


def opponent_name(playes_data):
    return playes_data[flip_color(my_color(playes_data))]["user"]["name"]


games = client.games.export_by_player(
    username="menisadi",
    perf_type="blitz",
    rated=True,
    analysed=True,
    opening=True,
    evals=True,
)

df = pd.DataFrame(
    columns=[
        "time",
        "opponent_name",
        "my_rating",
        "my_color",
        "result",
        "termination_type",
        "length_of_game",
        "my_average_centripawn_loss",
        "opponent_average_centripawn_loss",
        "opening",
    ]
)


for game in games:
    playes = game["players"]

    meni_color = my_color(playes)
    my_rating = game["players"][meni_color]["rating"]
    opponent_rating = game["players"][flip_color(meni_color)]["rating"]
    opponent = opponent_name(game["players"])
    time = game["createdAt"]
    termination = game["status"]
    winner = game["winner"]
    did_i_win = winner == meni_color
    openning = game["opening"]["name"]
    moves = game["moves"]
    game_length = len(moves.split(" "))
    my_accuracy = game["players"][meni_color]["analysis"]["acpl"]
    opponent_accuracy = game["players"][flip_color(meni_color)]["analysis"][
        "acpl"
    ]

    new_row = pd.DataFrame(
        [
            {
                "time": time,
                "opponent_name": opponent,
                "my_rating": my_rating,
                "my_color": meni_color,
                "result": winner,
                "did_I_win": did_i_win,
                "termination_type": termination,
                "length_of_game": game_length,
                "my_average_centripawn_loss": my_accuracy,
                "opponent_average_centripawn_loss": opponent_accuracy,
            }
        ]
    )

    pd.concat([df, new_row])
