from predict import predict, summarize, send_message
from update import update_averages, todays_games


avgDF = update_averages()
gamesDF = todays_games(avgDF)
predDF = predict(gamesDF)
summarize(predDF)
send_message("zach.harrison55@gmail.com")
send_message("Kjstudsrud@gmail.com")