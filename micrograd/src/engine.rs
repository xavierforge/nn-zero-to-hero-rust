use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

#[derive(Debug)]
pub struct Value(ValueRef);

type ValueRef = Rc<RefCell<ValueInner>>;

#[derive(Debug)]
struct ValueInner {
    data: f64,
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
            op: None,
            prev: Vec::new(),
            label: None,
        })))
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

    pub fn ptr(&self) -> *const () {
        Rc::as_ptr(&self.0) as *const ()
    }

    pub fn set_label(&self, label: String) {
        self.0.borrow_mut().label = Some(label)
    }

    pub fn label(&self) -> Option<String> {
        self.0.borrow().label.clone()
    }

    fn inner(&self) -> std::cell::Ref<'_, ValueInner> {
        self.0.borrow()
    }

    fn inner_mut(&self) -> std::cell::RefMut<'_, ValueInner> {
        self.0.borrow_mut()
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data() + rhs.data();

        let out = Value::new(data);
        out.inner_mut().op = Some("+");
        out.inner_mut().prev = vec![self.clone(), rhs.clone()];

        out
    }
}
