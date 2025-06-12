use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

type ValueRef = Rc<RefCell<Value>>;

#[derive(Debug)]
pub struct Value {
    data: f64,
    prev: Vec<ValueRef>,
    op: Option<&'static str>,
    label: Option<String>,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Self {
            data,
            prev: Vec::new(),
            op: None,
            label: None,
        }
    }

    pub fn data(&self) -> f64 {
        self.data
    }

    pub fn prev(&self) -> &[ValueRef] {
        &self.prev
    }

    pub fn op(&self) -> Option<&'static str> {
        self.op
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn set_label(&mut self, label: String) {
        self.label = Some(label)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            prev: vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))],
            op: Some("+"),
            label: None,
        }
    }
}
