Linear Regression Model using Burn in Rust

📌 Project Overview

This project implements a Linear Regression Model in Rust using the burn deep learning framework. The model learns to approximate a simple linear function:

using gradient descent and Mean Squared Error (MSE) loss. The project includes:
Generating Synthetic Data with noise.
Defining a Model using burn.
Training the Model to minimize loss.
Evaluating Performance with predictions and ASCII plots.

Getting Started

1️⃣ Prerequisites

Ensure you have the following installed:
Rust & Cargo → Install Rust
Burn Library (Machine Learning)
Rand Library (Generating Random Data)
Textplots Library (ASCII Graphs)

2️⃣ Setup the Project

Clone the repository and navigate to the project folder:

# Clone the repo (replace with your actual repo URL)
git clone https://github.com/your-username/linear-regression-burn.git
cd linear-regression-burn

3️⃣ Install Dependencies

Add the required dependencies to your Cargo.toml:

[dependencies]
burn = {version= "0.16.0” , features = ["wgpu” , "train”] }
burn-ndarray= "0.16.0”
rand = "0.9.0 ”
rgb = "0.8.50”
textplots = "0.8.6"

4️⃣ Run the Model

To train and test the linear regression model, run:

📜 Project Breakdown

1️⃣ Generating Synthetic Data
The dataset consists of (x, y) pairs where y = 2x + 1 + noise.
We introduce random noise to simulate real-world scenarios.

2️⃣ Defining the Model
A simple Linear Regression Model with one weight and one bias.
Forward pass: y_pred = weight * x + bias.
Uses Mean Squared Error (MSE) Loss for optimization.

3️⃣ Training the Model
Uses Stochastic Gradient Descent (SGD).
Updates weight and bias iteratively to minimize loss.


4️⃣ Evaluating & Plotting Results
Tests the trained model on new data.

🛠 Challenges Faced

Incorrect Cargo.toml Syntax: Had issues with curly quotes (” vs "), causing parsing errors. and i wasnt able to run the project and add the directories and fix the errors.

Missing link.exe Error: The Rust compiler couldn’t find link.exe (needed for MSVC). Wasnt still able to Solve the error even after by installing Visual Studio with C++ Build Tools.

Resources
• Rust Documentation: https://doc.rust-lang.org/
• Burn Library Documentation: https://docs.rs/burn/0.16.0/burn/
• GitHub Guide: https://docs.github.com/en/get-started
• Rust Rover Documentation: https://www.jetbrains.com/help/rust/
• LaTeX Documentation: https://www.latex-project.org/help/

Reflections & Learnings

How Much Help I Received?
AI Assistance (50%): Helped debug Cargo.toml errors, linking issues, and optimizing training.
Official Documentation (30%): Used extensively for understanding burn and Rust syntax.
Trial & Error (20%): Experimented with different hyperparameters (learning rate, epochs).