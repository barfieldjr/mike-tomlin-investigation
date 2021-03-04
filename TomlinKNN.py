import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing

data = pd.read_csv("NFL Play Mix.csv")
print(data.head())

le = preprocessing.LabelEncoder()
away_team = le.fit_transform(list(data["away_team"]))
side_of_field = le.fit_transform(list(data["side_of_field"]))
yardline_100 = le.fit_transform(list(data["yardline_100"]))
quarter_seconds_remaining = le.fit_transform(list(data["quarter_seconds_remaining"]))
half_seconds_remaining = le.fit_transform(list(data["half_seconds_remaining"]))
game_seconds_remaining = le.fit_transform(list(data["game_seconds_remaining"]))
game_half = le.fit_transform(list(data["game_half"]))
drive = le.fit_transform(list(data["drive"]))
qtr = le.fit_transform(list(data["qtr"]))
down = le.fit_transform(list(data["down"]))
ydstogo = le.fit_transform(list(data["ydstogo"]))
play = le.fit_transform(list(data["play_type"]))

predict = "play type"

x = list(zip(away_team,side_of_field,yardline_100,quarter_seconds_remaining,half_seconds_remaining,game_seconds_remaining,game_half,drive,qtr,down,ydstogo))
y = list(play)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_test)
acc = model.score(x_test, y_test)
print(acc)