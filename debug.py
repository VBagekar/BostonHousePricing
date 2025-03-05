import pickle

with open("regmodel.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))  # This should print the model type
