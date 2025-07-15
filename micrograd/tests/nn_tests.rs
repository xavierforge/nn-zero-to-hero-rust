use micrograd::engine::Value;
use micrograd::nn::{Layer, MLP, Neuron};

#[test]
fn test_neuron_forward() {
    let neuron = Neuron::new(3);
    let input = vec![Value::new(0.5), Value::new(-1.0), Value::new(2.0)];
    let output = neuron.forward(&input);
    assert!(output.data() >= -1.0 && output.data() <= 1.0);
}

#[test]
fn test_layer_forward() {
    let layer = Layer::new(3, 2);
    let input = vec![Value::new(0.5), Value::new(-1.0), Value::new(2.0)];
    let output = layer.forward(&input);
    assert_eq!(output.len(), 2);
    for val in output {
        assert!(val.data() >= -1.0 && val.data() <= 1.0);
    }
}

#[test]
fn test_mlp_forward() {
    let mlp = MLP::new(3, vec![4, 2, 1]);
    let input = vec![Value::new(0.5), Value::new(-1.0), Value::new(2.0)];
    let output = mlp.forward(&input);
    assert_eq!(output.len(), 1);
    for val in output {
        assert!(val.data() >= -1.0 && val.data() <= 1.0);
    }
}

#[test]
fn test_parameters() {
    let mlp = MLP::new(3, vec![4, 4, 1]);
    let params = mlp.parameters();
    assert_eq!(params.len(), 3 * 4 + 4 * 4 + 4 * 1 + 4 + 4 + 1); // weights + biases is 41
}
