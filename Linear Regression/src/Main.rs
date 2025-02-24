use burn::{
    data::dataset::{Dataset, InMemoryDataset},
    nn::{Linear, Loss, MSELoss},
    optim::{Gradients, Optimizer, SGD},
    tensor::{backend::Autodiff, Data, Tensor},
};
use rand::Rng;
use textplots::{Chart, Plot, Shape};

const DATASET_SIZE: usize = 100;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 500;

#[derive(Debug, Clone)]
struct LinearRegression {
    weight: Tensor<Autodiff<f32>, 1>,
    bias: Tensor<Autodiff<f32>, 1>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            weight: Tensor::random([1], -1.0, 1.0),
            bias: Tensor::random([1], -1.0, 1.0),
        }
    }

    fn forward(&self, x: &Tensor<Autodiff<f32>, 1>) -> Tensor<Autodiff<f32>, 1> {
        self.weight.clone() * x.clone() + self.bias.clone()
    }

    fn loss(&self, preds: &Tensor<Autodiff<f32>, 1>, targets: &Tensor<Autodiff<f32>, 1>) -> f32 {
        MSELoss::compute(preds, targets).into_scalar()
    }
}

// Generate synthetic dataset (y = 2x + 1 with noise)
fn generate_data() -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for _ in 0..DATASET_SIZE {
        let x: f32 = rng.gen_range(0.0..10.0);
        let noise: f32 = rng.gen_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + noise;
        x_vals.push(x);
        y_vals.push(y);
    }
    (x_vals, y_vals)
}

fn main() {
    let (x_vals, y_vals) = generate_data();

    let x_tensor = Tensor::<Autodiff<f32>, 1>::from_data(Data::from(x_vals.clone()));
    let y_tensor = Tensor::<Autodiff<f32>, 1>::from_data(Data::from(y_vals.clone()));

    let mut model = LinearRegression::new();
    let mut optimizer = SGD::new(LEARNING_RATE);

    // Training loop
    for epoch in 0..EPOCHS {
        let preds = model.forward(&x_tensor);
        let loss_value = model.loss(&preds, &y_tensor);

        let grads = Gradients::compute(&model.weight, &model.bias, |w, b| {
            let pred = w.clone() * x_tensor.clone() + b.clone();
            MSELoss::compute(&pred, &y_tensor)
        });

        optimizer.step(&mut model.weight, grads.get(&model.weight));
        optimizer.step(&mut model.bias, grads.get(&model.bias));

        if epoch % 50 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss_value);
        }
    }

    println!("\nFinal Model Parameters: Weight = {:.4}, Bias = {:.4}", 
        model.weight.clone().into_scalar(), 
        model.bias.clone().into_scalar()
    );

    // Evaluate and plot using textplots
    let predicted_y: Vec<f32> = x_vals.iter().map(|&x| 2.0 * x + 1.0).collect();
    println!("\nPlotting Results:");
    Chart::new(80, 20, 0.0, 10.0)
        .lineplot(&Shape::Bars(&x_vals, &predicted_y))
        .display();
}
