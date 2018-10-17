import twint

# Configure

users = ['RozeckaPL', 'GoTracz', 'martalempart', 'MichalakJerzy', 'KatarzynaObara', 'SutrykJacek']

c = twint.Config()
c.Store_csv = True
c.Custom = ["date", "time", "username", "tweet", "replies", "retweets", "likes", "user_rt"]
c.Output = "twitter.csv"

for user in users:
    c.Username = user
    a = twint.run.Profile(c)
