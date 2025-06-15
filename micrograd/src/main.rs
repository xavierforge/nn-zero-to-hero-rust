use micrograd::engine::Value;
use micrograd::trace_graph::draw_dot;

fn main() {
    let a = Value::new(3.0);
    let b = Value::new(5.0);
    let c = a + b;
    let d = Value::new(7.0);
    d.set_label(String::from("d"));
    let e = c + d;

    draw_dot(&e, "./test.svg");
}
