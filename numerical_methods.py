import numpy as np
import sympy as sp
from typing import Callable, Tuple, List


class NumericalMethods:
    """Clase principal que implementa métodos numéricos para encontrar raíces de ecuaciones."""
    
    def __init__(self):
        self.symbol = sp.symbols('x')

    def normalize_function_string(self, func_str: str) -> str:
        """Normaliza una cadena de función para ser compatible con SymPy."""
        s = func_str.strip()
        s = s.replace("∗", "*")
        s = s.replace("×", "*")
        s = s.replace("^", "**")
        s = s.replace(",", ".")
        return s

    def parse_function_expression(self, func_str: str) -> Tuple[Callable, sp.Expr]:
        """Convierte string de función a función numérica y expresión simbólica."""
        try:
            normalized = self.normalize_function_string(func_str)
            expr = sp.sympify(normalized)
            f = sp.lambdify(self.symbol, expr, 'numpy')
            return f, expr
        except Exception as e:
            normalized = self.normalize_function_string(func_str)
            raise ValueError(f"Función no válida: '{func_str}'. Intenta con: '{normalized}'.") from e
