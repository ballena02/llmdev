import calc
from logger import Logger

class Calculator:
    def __init__(self, logger: Logger):
        self.result = 0
        self.logger = logger

    def add(self, a):
        self.result = calc.add(self.result, a)
        self.logger.log(f"Add({a}): result = {self.result}")

    def subtract(self, a):
        self.result = calc.subtract(self.result, a)
        self.logger.log(f"Subtract({a}): result = {self.result}")

    def multiply(self, a):
        self.result = calc.multiply(self.result, a)
        self.logger.log(f"Multiply({a}): result = {self.result}")

    def divide(self, a):
        try:
            self.result = calc.divide(self.result, a)
            self.logger.log(f"Divide({a}): result = {self.result}")
        except ValueError as e:
            self.logger.log(f"Divide({a}): error = division by zero")
            raise ValueError("division error") from e

    def result(self):
        return self.result

    def reset(self):
        self.result = 0
        self.logger.log("Reset(): result = 0")
