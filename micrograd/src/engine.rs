use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

pub struct Value(Inner);

type Inner = Rc<RefCell<ValueInner>>;

struct ValueInner {
    data: f64,
    grad: f64,
    _backward: Option<Box<dyn Fn()>>,
    op: Option<&'static str>,
    prev: Vec<Value>,
    label: Option<String>,
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(Rc::clone(&self.0))
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            _backward: None,
            op: None,
            prev: Vec::new(),
            label: None,
        })))
    }

    fn from_binary_op<F>(
        lhs: Value,
        rhs: Value,
        op_str: &'static str,
        f: F,
        backward: Option<Box<dyn Fn()>>,
    ) -> Value
    where
        F: Fn(f64, f64) -> f64,
    {
        let value_inner = ValueInner {
            data: f(lhs.data(), rhs.data()),
            grad: 0.0,
            _backward: backward,
            prev: vec![lhs.clone(), rhs.clone()],
            op: Some(op_str),
            label: None,
        };

        Value(Rc::new(RefCell::new(value_inner)))
    }

    fn from_unary_op<F>(
        value: Value,
        op_str: &'static str,
        f: F,
        backward: Option<Box<dyn Fn()>>,
    ) -> Value
    where
        F: Fn(f64) -> f64,
    {
        let value_innter = ValueInner {
            data: f(value.data()),
            grad: 0.0,
            _backward: backward,
            prev: vec![value.clone()],
            op: Some(op_str),
            label: None,
        };

        Value(Rc::new(RefCell::new(value_innter)))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn op(&self) -> Option<&'static str> {
        self.0.borrow().op
    }

    pub fn prev(&self) -> Vec<Self> {
        self.0.borrow().prev.clone()
    }

    pub fn set_label(&self, label: String) {
        self.0.borrow_mut().label = Some(label)
    }

    pub fn label(&self) -> Option<String> {
        self.0.borrow().label.clone()
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad
    }

    pub fn backward(&self) {
        if let Some(ref func) = self.0.borrow()._backward {
            func()
        }
    }

    pub fn ptr(&self) -> *const () {
        Rc::as_ptr(&self.0) as *const ()
    }

    pub fn tanh(&self) -> Self {
        let t = self.data().tanh();
        let output = Value::from_unary_op(self.clone(), "tanh", |x| x.tanh(), None);
        let input_clone = self.clone();
        let output_clone = output.clone();
        let backward = move || {
            let grad = 1.0 - t * t;
            input_clone.set_grad(grad * output_clone.grad());
        };
        output.0.borrow_mut()._backward = Some(Box::new(backward));
        output
    }

    // fn inner(&self) -> std::cell::Ref<'_, ValueInner> {
    //     self.0.borrow()
    // }

    // fn inner_mut(&self) -> std::cell::RefMut<'_, ValueInner> {
    //     self.0.borrow_mut()
    // }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let lhs_clone = self.clone();
        let rhs_clone = rhs.clone();

        let output = Value::from_binary_op(self, rhs, "+", |a, b| a + b, None);
        let output_clone = output.clone();

        let backward = move || {
            lhs_clone.set_grad(1.0 * output_clone.grad());
            rhs_clone.set_grad(1.0 * output_clone.grad());
        };

        output.0.borrow_mut()._backward = Some(Box::new(backward));
        output
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let lhs_clone = self.clone();
        let rhs_clone = rhs.clone();

        let output = Value::from_binary_op(self, rhs, "*", |a, b| a * b, None);
        let output_clone = output.clone();

        let backward = move || {
            lhs_clone.set_grad(rhs_clone.data() * output_clone.grad());
            rhs_clone.set_grad(lhs_clone.data() * output_clone.grad());
        };

        output.0.borrow_mut()._backward = Some(Box::new(backward));
        output
    }
}
