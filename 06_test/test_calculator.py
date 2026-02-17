import pytest
from calculator import Calculator
from unittest.mock import MagicMock
from logger import Logger

@pytest.fixture
def calculator():
    # Loggerをモックに置き換える
    mock_logger = MagicMock(spec=Logger)
    calc = Calculator(logger=mock_logger)
    yield calc
    calc.reset()

# 各メソッドのテスト
@pytest.mark.parametrize("initial, add_value, expected", [
    (0, 5, 5),
    (5, 3, 8),
    (-5, -3, -8),
])
def test_add(calculator, initial, add_value, expected):
    calculator.add(initial)
    calculator.add(add_value)
    assert calculator.result == expected
    calculator.logger.log.assert_called()

@pytest.mark.parametrize("initial, sub_value, expected", [
    (10, 3, 7),
    (0, 5, -5),
    (-5, -5, 0),
])
def test_subtract(calculator, initial, sub_value, expected):
    calculator.add(initial)
    calculator.subtract(sub_value)
    assert calculator.result == expected
    calculator.logger.log.assert_called()

@pytest.mark.parametrize("initial, mul_value, expected", [
    (1, 3, 3),
    (0, 5, 0),
    (-2, 3, -6),
])
def test_multiply(calculator, initial, mul_value, expected):
    calculator.add(initial)
    calculator.multiply(mul_value)
    assert calculator.result == expected
    calculator.logger.log.assert_called()

@pytest.mark.parametrize("initial, div_value, expected", [
    (10, 2, 5),
    (9, 3, 3),
    (-6, -3, 2),
])
def test_divide(calculator, initial, div_value, expected):
    calculator.add(initial)
    calculator.divide(div_value)
    assert calculator.result == expected
    calculator.logger.log.assert_called()

def test_divide_by_zero(calculator):
    calculator.add(10)
    with pytest.raises(ValueError, match="division error"):
        calculator.divide(0)
    calculator.logger.log.assert_called_with("Divide(0): error = division by zero")

# 複雑な計算のテスト
def test_complex_calculation(calculator):
    calculator.add(2)
    calculator.multiply(3)
    calculator.subtract(1)
    calculator.divide(2)
    assert calculator.result == 2.5
    calculator.logger.log.assert_called()
