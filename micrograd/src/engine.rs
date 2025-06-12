use std::rc::Rc;

pub struct Value {
    data: f64,
    prev: Vec<Rc<Value>>,
    op: String,
    label: String,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            prev: vec![],
            op: String::new(),
            label: String::new(),
        }
    }

    pub fn data(&self) -> f64 {
        self.data
    }

    pub fn prev(&self) -> &[Rc<Value>] {
        &self.prev
    }
    pub fn op(&self) -> &str {
        &self.op
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}
