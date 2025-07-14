use crate::engine::Value;
use rand::Rng;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: u32) -> Self {
        let mut rng = rand::rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.random_range(-1.0..=1.0)))
            .collect();
        let b = Value::new(rng.random_range(-1.0..=1.0));
        Neuron { w, b }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        // w * x + b
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = act + wi.clone() * xi.clone();
        }
        act.tanh()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: u32, nout: u32) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}
