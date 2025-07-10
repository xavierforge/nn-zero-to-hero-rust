use micrograd::engine::Value;
use micrograd::trace_graph::draw_dot;

fn main() {
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

    draw_dot(&o, "./graph.svg");
}
