# Movie-Recommender-System-using-AutoEncoders
Enhance your movie-watching experience with Collaborative Movie Recommendations using AutoEncoders. Trained on the MovieLens dataset, this PyTorch-powered system predicts unrated movie ratings with an impressive accuracy of ±0.9493. Interactively input your ratings for personalized Movie Suggestions. 


## Overview

This collaborative filtering project employs AutoEncoders to predict unrated movie ratings, providing accurate movie recommendations. Trained on the MovieLens dataset, the AutoEncoder model achieves a remarkable accuracy of ±0.9493, showcasing its effectiveness in personalized movie suggestions.

## Key Features

- **Data Processing:**
  - Utilizes MovieLens dataset, comprising movies, users, and ratings.
  - Prepares training and test sets for evaluating the model's performance.

- **AutoEncoder Model:**
  - Implements a Stacked AutoEncoder (SAE) architecture using PyTorch.
  - Employs Mean Squared Error (MSE) loss and RMSprop optimizer for training.

- **Training and Evaluation:**
  - Trains the AutoEncoder model on the training set, optimizing for unrated movie predictions.
  - Evaluates model performance on the test set, achieving an accuracy of ±0.9493.

- **User Interaction:**
  - Allows users to input movie ratings interactively for personalized recommendations.
  - Recommends movies based on user input and predicted ratings.

## Usage

1. **Training the AutoEncoder:**
   - Run the provided script to train the AutoEncoder model on the MovieLens dataset.

2. **User Interaction:**
   - Interact with the system by inputting movie ratings.
   - Receive personalized movie recommendations based on the provided ratings.

3. **Adjusting Threshold:**
   - Fine-tune the recommendation threshold for controlling the number of recommended movies.

## Dependencies

- Python 
- PyTorch
- NumPy
- pandas

## Getting Started

1. **Clone the Repository:**
   - Clone this repository to your local machine using the command: `git clone https://github.com/yourusername/movie-recommendations.git`

2. **Install Dependencies:**
   - Install the required dependencies by running: `pip install -r requirements.txt`

3. **Run Training Script:**
   - Execute the training script to train the AutoEncoder: `python train_autoencoder.py`

4. **Interact with the System:**
   - Run the interactive script to input movie ratings and receive personalized recommendations: `python interact.py`


## Acknowledgments

- This project was inspired by collaborative filtering techniques in recommendation systems.
- Special thanks to the MovieLens dataset for providing a rich source of movie-related data.

Feel free to explore, contribute, and adapt the project based on your preferences. Feedback and contributions are highly appreciated!
