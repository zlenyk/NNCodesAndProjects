import pickle
from lstm import generate_text
model = pickle.load( open( "model.p", "rb" ) )

generate_text(model, "Jam jest Jacek")
