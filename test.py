from nltk import tokenize

p = "Good morning Dr. Adams. The patient is waiting for you in room number 3."
print(tokenize.sent_tokenize(p))
