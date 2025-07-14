use micrograd::engine::Value;
use micrograd::nn::MLP;
use micrograd::trace_graph::draw_dot;

fn draw_value_example() {
    // inputs x1, x2
    let x1 = Value::new(2.0);
    x1.set_label("x1".to_string());
    let x2 = Value::new(0.0);
    x2.set_label("x2".to_string());
    // weights w1, w2
    let w1 = Value::new(-3.0);
    w1.set_label("w1".to_string());
    let w2 = Value::new(1.0);
    w2.set_label("w2".to_string());
    // bias b
    let b = Value::new(6.881373587019543);
    b.set_label("b".to_string());
    // linear combination
    let x1w1 = x1.clone() * w1.clone();
    x1w1.set_label("x1 * w1".to_string());
    let x2w2 = x2.clone() * w2.clone();
    x2w2.set_label("x2 * w2".to_string());
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    x1w1x2w2.set_label("x1 * w1 + x2 * w2".to_string());
    // final output n = x1w1 + x2w2 + b
    let n = x1w1x2w2.clone() + b.clone();
    n.set_label("n".to_string());
    let o = n.tanh();
    o.set_label("o".to_string());

    o.backward();

    draw_dot(&o, "./value_example.svg");
}

fn draw_mlp_example() {
    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    let mlp = MLP::new(3, vec![4, 4, 1]);
    let y_pred = xs
        .iter()
        .map(|x| mlp.forward(x))
        .flatten()
        .collect::<Vec<_>>();
    let loss = ys
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| (y.clone() - y_hat.clone()).powi(2))
        .reduce(|acc, x| acc + x) // does not implement Sum trait so we use reduce
        .unwrap();

    loss.backward();
    draw_dot(&loss, "./mlp_example.svg");
}

fn main() {
    draw_value_example();
    draw_mlp_example();
}
