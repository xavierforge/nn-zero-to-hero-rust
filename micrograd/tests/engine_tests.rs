use micrograd::engine::Value;

#[test]
fn test_value_creation() {
    let value = Value::new(3.0);

    assert_eq!(value.data(), 3.0, "Expected value.data to be 3.0");
    assert_eq!(value.prev().len(), 0, "Expected value.prev to be empty");
    assert_eq!(value.op(), "", "Expected value.op to be None");
    assert_eq!(value.label(), "", "Expected value.label to be None");
}
