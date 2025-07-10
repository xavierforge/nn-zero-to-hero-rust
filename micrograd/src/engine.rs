use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Add, Mul};
use std::rc::Rc;

// ============================================================================
// Core Types and Data Structures
// ============================================================================

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

// ============================================================================
// Basic Implementations
// ============================================================================

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(Rc::clone(&self.0))
    }
}

impl Value {
    // ========================================================================
    // Constructors
    // ========================================================================

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

    // ========================================================================
    // Internal Operation Builders
    // ========================================================================

    fn unary_op_with_backward<F, B>(input: Value, op_str: &'static str, op_fn: F, bw_fn: B) -> Value
    where
        F: Fn(f64) -> f64,
        B: Fn(Value, Value) -> Box<dyn Fn()>,
    {
        let output = Value(Rc::new(RefCell::new(ValueInner {
            data: op_fn(input.data()),
            grad: 0.0,
            _backward: None,
            op: Some(op_str),
            prev: vec![input.clone()],
            label: None,
        })));
        output.0.borrow_mut()._backward = Some(bw_fn(input, output.clone()));
        output
    }

    fn binary_op_with_backward<F, B>(
        lhs: Value,
        rhs: Value,
        op_str: &'static str,
        op_fn: F,
        bw_fn: B,
    ) -> Value
    where
        F: Fn(f64, f64) -> f64,
        B: Fn(Value, Value, Value) -> Box<dyn Fn()>,
    {
        let output = Value(Rc::new(RefCell::new(ValueInner {
            data: op_fn(lhs.data(), rhs.data()),
            grad: 0.0,
            _backward: None,
            op: Some(op_str),
            prev: vec![lhs.clone(), rhs.clone()],
            label: None,
        })));
        output.0.borrow_mut()._backward = Some(bw_fn(lhs, rhs, output.clone()));
        output
    }

    // ========================================================================
    // Public Accessors
    // ========================================================================

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn op(&self) -> Option<&'static str> {
        self.0.borrow().op
    }

    pub fn prev(&self) -> Vec<Self> {
        self.0.borrow().prev.clone()
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn label(&self) -> Option<String> {
        self.0.borrow().label.clone()
    }

    pub fn ptr(&self) -> *const () {
        Rc::as_ptr(&self.0) as *const ()
    }

    // ========================================================================
    // Mutators
    // ========================================================================

    pub fn set_label(&self, label: String) {
        self.0.borrow_mut().label = Some(label)
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad
    }

    // ========================================================================
    // Backpropagation
    // ========================================================================

    pub fn backward(&self) {
        let mut seen: HashMap<*const (), bool> = HashMap::new();
        let mut topo: Vec<Value> = Vec::new();

        fn build_topo(v: &Value, seen: &mut HashMap<*const (), bool>, topo: &mut Vec<Value>) {
            let ptr = v.ptr();
            if !seen.contains_key(&ptr) {
                seen.insert(ptr, true);
                for prev in v.prev() {
                    build_topo(&prev, seen, topo);
                }
                topo.push(v.clone());
            }
        }
        build_topo(self, &mut seen, &mut topo);
        self.set_grad(1.0);
        topo.reverse();
        for v in topo {
            if let Some(ref func) = v.0.borrow()._backward {
                func()
            }
        }
    }

    // ========================================================================
    // Mathematical Operations
    // ========================================================================

    pub fn tanh(&self) -> Self {
        let t = self.data().tanh();
        Value::unary_op_with_backward(
            self.clone(),
            "tanh",
            |x| x.tanh(),
            |input, output| {
                Box::new(move || {
                    let grad = 1.0 - t * t;
                    input.set_grad(grad * output.grad());
                })
            },
        )
    }
}

// ============================================================================
// Operator Trait Implementations
// ============================================================================

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::binary_op_with_backward(
            self,
            rhs,
            "+",
            |a, b| a + b,
            |lhs, rhs, output| {
                Box::new(move || {
                    lhs.set_grad(1.0 * output.grad());
                    rhs.set_grad(1.0 * output.grad());
                })
            },
        )
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value::binary_op_with_backward(
            self,
            rhs,
            "*",
            |a, b| a * b,
            |lhs, rhs, output| {
                Box::new(move || {
                    lhs.set_grad(rhs.data() * output.grad());
                    rhs.set_grad(lhs.data() * output.grad());
                })
            },
        )
    }
}
