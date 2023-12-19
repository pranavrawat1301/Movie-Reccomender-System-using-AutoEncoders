import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        self.batch_norm1 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.001, weight_decay=0.5)

# Training the SAE
nb_epoch = 400
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input_tensor = Variable(training_set[id_user]).unsqueeze(0)
        target = input_tensor.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input_tensor)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
    
    
    
    
    
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input_tensor = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input_tensor)
        target.require_grad = False
        output[0,target == 0] = 0
        loss = criterion(output[0], target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))





# Function to get recommended movies
def get_recommendations(ratings, threshold=3):
    input_movies = np.zeros(nb_movies)
    input_movies[ratings[:, 0] - 1] = ratings[:, 1]  # User ratings
    input_movies = torch.FloatTensor([input_movies])
    reconstructed_movies = sae(input_movies)
    predicted_ratings = reconstructed_movies.squeeze().detach().numpy()
    unrated_movies = np.where(input_movies.squeeze().numpy() == 0)[0]
    recommended_movie_ids = unrated_movies[predicted_ratings[unrated_movies] > threshold]
    return recommended_movie_ids + 1

# Interactive part
user_ratings = []
while True:
    try:
        user_input = input("Enter a movie title (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        movie_id = movies[movies[1] == user_input][0].values
        if len(movie_id) == 0:
            print(f"Movie '{user_input}' not found.")
            continue
        rating = float(input(f"Rate the movie '{user_input}' on a scale of 1 to 5: "))
        if rating < 1 or rating > 5:
            print("Invalid rating. Please enter a rating between 1 and 5.")
            continue
        user_ratings.append([int(movie_id), rating])
    except KeyboardInterrupt:
        print("\nExiting...")
        break

user_ratings = np.array(user_ratings)
print("User ratings:")
print(user_ratings)

# Get movie recommendations based on user ratings
recommended_movies = get_recommendations(user_ratings)

# Display recommended movies
print("\nRecommended movies:")
for movie_id in recommended_movies:
    movie_title = movies[movies[0] == movie_id][1].values[0]
    print(f"{movie_title} (Movie ID: {movie_id})")


