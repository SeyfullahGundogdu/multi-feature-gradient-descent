#![allow(non_snake_case)]
use std::fs;

fn main() {
    let (X_normalized, Y) = read_file();
    let (m, n) = (X_normalized.len(), X_normalized[0].len());
    // y_hat(i) = w * x + b; where w and x are Vector of f64
    // weights(coefficients)
    let X_train: Vec<Vec<f64>> = X_normalized[0..404].into();
    let X_predict: Vec<Vec<f64>> = X_normalized[404..m].into();

    let Y_train: Vec<f64> = Y[0..404].into();
    let Y_predict: Vec<f64> = Y[404..Y.len()].into();

    let w = vec![0f64; n];
    let b = 0f64;
    let a = 0.0001;

    let (w, b) = gradient_descent(X_train.as_slice(), &Y_train, &w, b, a, 100000);
    let cost = j_wb(&w, &X_predict, &Y_predict, b);
    println!("w vector: {:.4?}\nb: {:.4}\ncost: {:.4}", w, b, cost);
}

fn read_file() -> (Vec<Vec<f64>>, Vec<f64>) {
    //read file to a string, return error if not found.
    let contents = fs::read_to_string("housing.csv")
        .expect("housing.csv not found. Try running \"cargo run --release\" in the project's root directory.");
    println!("Splitting dataset and using a portion of it as training set.");
    //split the string in chunks using new line as delimiter.
    let lines: Vec<String> = contents.split('\n').map(|s| s.to_string()).collect();
    //float vectors to store numbers
    let mut X: Vec<Vec<f64>> = vec![];
    let mut Y: Vec<f64> = vec![];
    //convert each line into vectors of floats, use first 13 elements for X (features) vector, last element is our target. 
    lines.iter().for_each(|line| {
        let t = line
            .split_ascii_whitespace()
            .map(|t| t.parse::<f64>().unwrap_or_default())
            .collect::<Vec<f64>>();
        X.push(t[0..(t.len() - 1)].to_owned());
        Y.push(t[t.len() - 1]);
    });

    //normalize our X vector using vector scaling.
    let mut X_max_min = vec![0f64; X[0].len()];
    let mut X_average: Vec<f64> = vec![0f64; X[0].len()];
    for j in 0..X[0].len() {
        X_max_min[j] = X
            .iter()
            .map(|x| {
                X_average[j] += x[j];
                x[j]
            })
            .reduce(f64::max)
            .unwrap()
            - X.iter().map(|x| x[j]).reduce(f64::min).unwrap();
    }
    for elem in X_average.iter_mut() {
        *elem /= X.len() as f64;
    }

    let X_normalized: Vec<Vec<f64>> = X
        .iter()
        .map(|x| {
            x.iter()
                .enumerate()
                .map(|(i, x_i)| (x_i - X_average[i]) / X_max_min[i])
                .collect()
        })
        .collect();
    (X_normalized, Y)
}
/// model:
/// f [w,b](x) = w * x + b, where w and b are weights(coefficients)
/// for each element that w and x have, we multiply them and get their sum, plus b if we have bias
fn f_wb(w: &[f64], x: &[f64], b: f64) -> f64 {
    w.iter().enumerate().map(|(i, w_i)| w_i * x[i]).sum::<f64>() + b
}

/// Squared Error Cost Function: calculates the squared sum of each x row's
/// predicted value (prediction) and subtracts it from known Y values(target)
fn j_wb(w: &[f64], X: &[Vec<f64>], Y: &[f64], b: f64) -> f64 {
    let mut cost = 0f64;
    X.iter()
        .enumerate()
        .for_each(|(i, x)| cost += (f_wb(w, x, b) - Y[i]).powi(2));
    cost /= (2 * X.len()) as f64;
    cost
}

/// Computes the gradient for linear regression
fn compute_gradient(w: &[f64], X: &[Vec<f64>], Y: &[f64], b: f64) -> (Vec<f64>, f64) {
    let mut dj_dw = vec![0f64; w.len()];
    let mut dj_db = 0f64;
    let m = X.len();
    for i in 0..m {
        let fwb = f_wb(w, &X[i], b) - Y[i];
        for (j, elem) in dj_dw.iter_mut().enumerate() {
            *elem += fwb * X[i][j];
        }
        dj_db += fwb
    }
    let m = m as f64;
    dj_dw.iter_mut().for_each(|x| *x /= m);
    dj_db /= m;
    (dj_dw, dj_db)
}

fn gradient_descent(
    X: &[Vec<f64>],
    Y: &[f64],
    w: &Vec<f64>,
    b: f64,
    a: f64,
    num_iters: usize,
) -> (Vec<f64>, f64) {
    let mut w = w.to_owned();
    let mut b = b;

    for _ in 0..num_iters {
        let (dj_dw, dj_db) = compute_gradient(&w, X, Y, b);
        w.iter_mut()
            .enumerate()
            .for_each(|(i, w_i)| *w_i -= dj_dw[i] * a);
        b -= a * dj_db;
    }
    (w, b)
}
