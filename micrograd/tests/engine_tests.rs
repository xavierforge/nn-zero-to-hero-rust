use micrograd::engine::Value;

#[test]
fn test_value_creation() {
    let value = Value::new(3.0);

    assert_eq!(value.data(), 3.0, "Expected value.data to be 3.0");
    assert!(value.prev().is_empty(), "Expected value.prev to be empty");
    assert_eq!(value.op(), None, "Expected value.op to be None");
}

#[test]
fn test_value_addition() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);

    let c = a + b;

    assert_eq!(c.data(), 5.0, "Expected c.data to be 5.0");
    assert_eq!(c.op(), Some("+"), "Expected c.op to be '+'");
    assert_eq!(c.prev().len(), 2, "Expected c.prev to contain two parents");
    assert_eq!(c.prev()[0].data(), 2.0);
    assert_eq!(c.prev()[1].data(), 3.0);
}

#[test]
fn test_label_setting() {
    let value = Value::new(1.0);
    value.set_label(String::from("Cool"));

    assert_eq!(
        value.label(),
        Some(String::from("Cool")),
        "Expected value.label to be Cool"
    )
}
