use micrograd::engine::Value;
use micrograd::nn::Neuron;

#[test]
fn test_neuron_forward() {
    let neuron = Neuron::new(3);
    let input = vec![Value::new(0.5), Value::new(-1.0), Value::new(2.0)];
    let output = neuron.forward(&input);
    assert!(output.data() >= -1.0 && output.data() <= 1.0);
}
