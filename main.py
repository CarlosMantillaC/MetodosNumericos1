import numpy as np
import sympy as sp
from typing import Callable, Tuple, List
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import subprocess
import tempfile
import webbrowser
import customtkinter as ctk


class MetodosNumericos:
    def __init__(self):
        self.x = sp.symbols('x')

    def _normalize_func_str(self, func_str: str) -> str:
        s = func_str.strip()
        s = s.replace("∗", "*")
        s = s.replace("×", "*")
        s = s.replace("^", "**")
        s = s.replace(",", ".")
        return s

    def parse_function(self, func_str: str) -> Callable:
        """Convierte string de función a función numérica"""
        try:
            normalized = self._normalize_func_str(func_str)
            expr = sp.sympify(normalized)
            f = sp.lambdify(self.x, expr, 'numpy')
            return f, expr
        except Exception as e:
            normalized = self._normalize_func_str(func_str)
            raise ValueError(f"Función no válida: '{func_str}'. Intenta con: '{normalized}'.") from e

    def punto_fijo(
            self,
            func_str: str,
            x0: float,
            tol: float = 1e-6,
            max_iter: int = 100,
            criterio: str = 'error',
            stop_iter: int | None = None,
            blow_up_limit: float = 1e12,
            nondecrease_patience: int = 8,
    ) -> Tuple[List, str]:
        """Método de aproximaciones sucesivas (punto fijo)"""
        f, expr = self.parse_function(func_str)

        iteraciones = []
        x_actual = x0
        error = float('inf')
        status = "max_iter"
        nondec_count = 0
        prev_error = None

        for i in range(max_iter):
            x_siguiente = f(x_actual)

            if not np.isfinite(x_siguiente):
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'x_siguiente': x_siguiente,
                    'error': float('inf')
                })
                break

            if abs(x_siguiente) > blow_up_limit:
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'x_siguiente': x_siguiente,
                    'error': float('inf')
                })
                break

            # Calcular error real para detección de divergencia
            error_real = abs(x_siguiente - x_actual)
            
            iteraciones.append({
                'iter': i + 1,
                'x_actual': x_actual,
                'x_siguiente': x_siguiente,
                'error': error_real if criterio == 'error' else i + 1
            })

            if criterio == 'error':
                if error_real < tol:
                    status = "convergio"
                    break

                if prev_error is not None:
                    if error_real >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error_real

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
            elif criterio == 'iteracion':
                # Cuando el criterio es iteracion, usamos el error real para detectar divergencia
                if prev_error is not None:
                    # Si el error es cero o muy pequeño, no lo contamos como no-disminución (convergencia perfecta)
                    if error_real <= 1e-15:  # Umbral para convergencia numérica
                        nondec_count = 0  # Reiniciar contador, convergencia perfecta
                    elif error_real >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error_real

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
                
                if (i + 1) >= max_iter:
                    status = "iteraciones"
                    break

            x_actual = x_siguiente
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        latex_output = self._latex_punto_fijo(func_str, x0, tol, max_iter, criterio, iteraciones, status)
        return iteraciones, latex_output

    def newton_raphson(self, func_str: str, x0: float, tol: float = 1e-6, max_iter: int = 100,
                       criterio: str = 'error', blow_up_limit: float = 1e12,
                       nondecrease_patience: int = 8, deriv_eps: float = 1e-14) -> Tuple[List, str]:
        """Método de Newton-Raphson"""
        f, expr = self.parse_function(func_str)
        f_prime = sp.diff(expr, self.x)
        df = sp.lambdify(self.x, f_prime, 'numpy')

        iteraciones = []
        x_actual = x0
        error = float('inf')
        status = "max_iter"
        nondec_count = 0
        prev_error = None

        for i in range(max_iter):
            fx = f(x_actual)
            dfx = df(x_actual)

            if np.iscomplexobj(fx) or np.iscomplexobj(dfx):
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            if not np.isfinite(fx) or not np.isfinite(dfx):
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            if abs(dfx) < deriv_eps:
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': x_actual,
                    'error': float('inf')
                })
                break

            x_siguiente = x_actual - fx / dfx

            if np.iscomplexobj(x_siguiente):
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': x_siguiente,
                    'error': float('inf')
                })
                break

            if not np.isfinite(x_siguiente) or abs(x_siguiente) > blow_up_limit:
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_actual': x_actual,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': x_siguiente,
                    'error': float('inf')
                })
                break

            # Calcular error real para detección de divergencia
            error_real = abs(x_siguiente - x_actual)
            
            iteraciones.append({
                'iter': i + 1,
                'x_actual': x_actual,
                'fx': fx,
                'dfx': dfx,
                'x_siguiente': x_siguiente,
                'error': error_real if criterio == 'error' else i + 1
            })

            if criterio == 'error':
                if error_real < tol:
                    status = "convergio"
                    break

                if prev_error is not None:
                    if error_real >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error_real

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
            elif criterio == 'iteracion':
                # Cuando el criterio es iteracion, usamos el error real para detectar divergencia
                if prev_error is not None:
                    # Si el error es cero o muy pequeño, no lo contamos como no-disminución (convergencia perfecta)
                    if error_real <= 1e-15:  # Umbral para convergencia numérica
                        nondec_count = 0  # Reiniciar contador, convergencia perfecta
                    elif error_real >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error_real

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
                
                if (i + 1) >= max_iter:
                    status = "iteraciones"
                    break

            x_actual = x_siguiente
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        latex_output = self._latex_newton(func_str, x0, tol, max_iter, criterio, iteraciones, status)
        return iteraciones, latex_output

    def secante(self, func_str: str, x0: float, x1: float, tol: float = 1e-6,
                max_iter: int = 100, criterio: str = 'error', blow_up_limit: float = 1e12,
                nondecrease_patience: int = 8) -> Tuple[List, str]:
        """Método de la secante"""
        f, expr = self.parse_function(func_str)

        iteraciones = []
        x_anterior = x0
        x_actual = x1
        error = float('inf')
        status = "max_iter"
        prev_error = None
        nondec_count = 0

        for i in range(max_iter):
            fx_anterior = f(x_anterior)
            fx_actual = f(x_actual)

            if np.iscomplexobj(fx_anterior) or np.iscomplexobj(fx_actual):
                status = "divergio"
                iteraciones.append({
                    'iter': i + 1,
                    'x_anterior': x_anterior,
                    'x_actual': x_actual,
                    'fx_anterior': fx_anterior,
                    'fx_actual': fx_actual,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            # Calcular x_siguiente (manejando división por cero)
            if fx_actual - fx_anterior == 0:
                if criterio == 'iteracion':
                    # Cuando el criterio es iteracion y hay división por cero,
                    # simplemente continuamos con el valor actual
                    x_siguiente = x_actual
                    error_real = 0.0  # No hay cambio, error es cero
                else:
                    status = "divergio"
                    iteraciones.append({
                        'iter': i + 1,
                        'x_anterior': x_anterior,
                        'x_actual': x_actual,
                        'fx_anterior': fx_anterior,
                        'fx_actual': fx_actual,
                        'x_siguiente': float('nan'),
                        'error': float('inf')
                    })
                    break
            else:
                x_siguiente = x_actual - fx_actual * (x_actual - x_anterior) / (fx_actual - fx_anterior)

                if np.iscomplexobj(x_siguiente):
                    status = "divergio"
                    iteraciones.append({
                        'iter': i + 1,
                        'x_anterior': x_anterior,
                        'x_actual': x_actual,
                        'fx_anterior': fx_anterior,
                        'fx_actual': fx_actual,
                        'x_siguiente': x_siguiente,
                        'error': float('inf')
                    })
                    break

                if not np.isfinite(x_siguiente) or abs(x_siguiente) > blow_up_limit:
                    status = "divergio"
                    iteraciones.append({
                        'iter': i + 1,
                        'x_anterior': x_anterior,
                        'x_actual': x_actual,
                        'fx_anterior': fx_anterior,
                        'fx_actual': fx_actual,
                        'x_siguiente': x_siguiente,
                        'error': float('inf')
                    })
                    break

                # Calcular error real para detección de divergencia
                error_real = abs(x_siguiente - x_actual)
            
            if criterio == 'error':
                error = error_real
            elif criterio == 'iteracion':
                error = i + 1

            iteraciones.append({
                'iter': i + 1,
                'x_anterior': x_anterior,
                'x_actual': x_actual,
                'fx_anterior': fx_anterior,
                'fx_actual': fx_actual,
                'x_siguiente': x_siguiente,
                'error': error
            })

            if criterio == 'error':
                if error < tol:
                    status = "convergio"
                    break

                if prev_error is not None:
                    if error >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
            elif criterio == 'iteracion':
                # Cuando el criterio es iteracion, usamos el error real para detectar divergencia
                if prev_error is not None:
                    # Si el error es cero o muy pequeño, no lo contamos como no-disminución (convergencia perfecta)
                    if error_real <= 1e-15:  # Umbral para convergencia numérica
                        nondec_count = 0  # Reiniciar contador, convergencia perfecta
                    elif error_real >= prev_error:
                        nondec_count += 1
                    else:
                        nondec_count = 0
                prev_error = error_real

                if nondec_count >= nondecrease_patience:
                    status = "divergio"
                    break
                
                if (i + 1) >= max_iter:
                    status = "iteraciones"
                    break

            x_anterior = x_actual
            x_actual = x_siguiente
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        latex_output = self._latex_secante(func_str, x0, x1, tol, max_iter, criterio, iteraciones, status)
        return iteraciones, latex_output

    def _latex_punto_fijo(self, func_str: str, x0: float, tol: float, max_iter: int,
                          criterio: str, iteraciones: List, status: str) -> str:
        func_latex = sp.latex(sp.sympify(self._normalize_func_str(func_str)))
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de Aproximaciones Sucesivas (Punto Fijo)}}

\\textbf{{Función:}} $g(x) = {func_latex}$

\\textbf{{Valor inicial:}} $x_0 = {x0}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        def _fmt_num(v, kind: str) -> str:
            try:
                if v is None:
                    return "-"
                if isinstance(v, str) and ("j" in v or "J" in v):
                    s = v.strip().strip("()")
                    s = s.replace("J", "j")
                    s = s.replace("j", "")
                    return s + "\\,i"
                if isinstance(v, (complex, np.complexfloating)) or np.iscomplexobj(v):
                    c = complex(v)
                    sign = "+" if c.imag >= 0 else "-"
                    if kind == "float":
                        return f"{c.real:.6f} {sign} {abs(c.imag):.2e}\\,i"
                    if kind == "sci":
                        return f"{c.real:.2e} {sign} {abs(c.imag):.2e}\\,i"
                    return f"{c.real} {sign} {abs(c.imag)} i"
                if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                    return "\\infty" if v > 0 else "-\\infty"
                if kind == "float":
                    return f"{float(v):.6f}"
                if kind == "sci":
                    return f"{float(v):.2e}"
                return str(v)
            except Exception:
                return str(v)

        for it in iteraciones:
            latex += (
                f"{it['iter']} & ${_fmt_num(it['x_actual'], 'float')}$ & ${_fmt_num(it['x_siguiente'], 'float')}$ & ${_fmt_num(it['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iteraciones:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${_fmt_num(iteraciones[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${_fmt_num(iteraciones[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex

    def _latex_newton(self, func_str: str, x0: float, tol: float, max_iter: int,
                      criterio: str, iteraciones: List, status: str) -> str:
        func_latex = sp.latex(sp.sympify(self._normalize_func_str(func_str)))
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de Newton-Raphson}}

\\textbf{{Función:}} $f(x) = {func_latex}$

\\textbf{{Valor inicial:}} $x_0 = {x0}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$f'(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$f'(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        def _fmt_num(v, kind: str) -> str:
            try:
                if v is None:
                    return "-"
                if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                    return "\\infty" if v > 0 else "-\\infty"
                if kind == "float":
                    return f"{float(v):.6f}"
                if kind == "sci":
                    return f"{float(v):.2e}"
                return str(v)
            except Exception:
                return str(v)

        for it in iteraciones:
            latex += (
                f"{it['iter']} & ${_fmt_num(it['x_actual'], 'float')}$ & ${_fmt_num(it['fx'], 'float')}$ & ${_fmt_num(it['dfx'], 'float')}$ & ${_fmt_num(it['x_siguiente'], 'float')}$ & ${_fmt_num(it['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iteraciones:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${_fmt_num(iteraciones[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${_fmt_num(iteraciones[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex

    def _latex_secante(self, func_str: str, x0: float, x1: float, tol: float, max_iter: int,
                       criterio: str, iteraciones: List, status: str) -> str:
        func_latex = sp.latex(sp.sympify(self._normalize_func_str(func_str)))
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de la Secante}}

\\textbf{{Función:}} $f(x) = {func_latex}$

\\textbf{{Valores iniciales:}} $x_0 = {x0}$, $x_1 = {x1}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i-1}}$}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i-1}})$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i-1}}$}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i-1}})$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        def _fmt_num(v, kind: str) -> str:
            try:
                if v is None:
                    return "-"
                if isinstance(v, str) and ("j" in v or "J" in v):
                    s = v.strip().strip("()")
                    s = s.replace("J", "j")
                    s = s.replace("j", "")
                    return s + "\\,i"
                if isinstance(v, (complex, np.complexfloating)) or np.iscomplexobj(v):
                    c = complex(v)
                    sign = "+" if c.imag >= 0 else "-"
                    if kind == "float":
                        return f"{c.real:.6f} {sign} {abs(c.imag):.2e}\\,i"
                    if kind == "sci":
                        return f"{c.real:.2e} {sign} {abs(c.imag):.2e}\\,i"
                    return f"{c.real} {sign} {abs(c.imag)} i"
                if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                    return "\\infty" if v > 0 else "-\\infty"
                if kind == "float":
                    return f"{float(v):.6f}"
                if kind == "sci":
                    return f"{float(v):.2e}"
                return str(v)
            except Exception:
                return str(v)

        for it in iteraciones:
            latex += (
                f"{it['iter']} & ${_fmt_num(it['x_anterior'], 'float')}$ & ${_fmt_num(it['x_actual'], 'float')}$ "
                f"& ${_fmt_num(it['fx_anterior'], 'float')}$ & ${_fmt_num(it['fx_actual'], 'float')}$ "
                f"& ${_fmt_num(it['x_siguiente'], 'float')}$ & ${_fmt_num(it['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iteraciones:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${_fmt_num(iteraciones[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${_fmt_num(iteraciones[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex


class MetodosNumericosGUI:
    def __init__(self):
        # Configurar customtkinter
        ctk.set_appearance_mode("dark")  # Opciones: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Opciones: "blue", "green", "dark-blue"
        
        self.root = ctk.CTk()
        self.root.title("Métodos Numéricos")
        self.root.geometry("800x600")  # Tamaño inicial

        self.mn = MetodosNumericos()

        self.method_var = ctk.StringVar(value="punto_fijo")
        self.func_var = ctk.StringVar(value="0.5*x+1.0/x")
        self.x0_var = ctk.StringVar(value="1")
        self.x1_var = ctk.StringVar(value="2")
        self.tol_var = ctk.StringVar(value="1e-6")
        self.max_iter_var = ctk.StringVar(value="100")
        self.criterio_var = ctk.StringVar(value="error")

        self._build_ui()
        self._on_method_change()

    def _build_ui(self):
        # Frame principal
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Frame de controles
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(pady=10, padx=10, fill="x")
        
        # Método
        ctk.CTkLabel(control_frame, text="Método:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        method_cb = ctk.CTkComboBox(
            control_frame,
            variable=self.method_var,
            values=["punto_fijo", "newton_raphson", "secante"],
            width=200
        )
        method_cb.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))
        method_cb.bind("<<ComboboxSelected>>", lambda e: self._on_method_change())
        
        # Función
        ctk.CTkLabel(control_frame, text="Función:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        func_entry = ctk.CTkEntry(control_frame, textvariable=self.func_var, width=200)
        func_entry.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # x0
        ctk.CTkLabel(control_frame, text="x0:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        x0_entry = ctk.CTkEntry(control_frame, textvariable=self.x0_var, width=200)
        x0_entry.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # x1
        ctk.CTkLabel(control_frame, text="x1 (secante):", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        self.x1_entry = ctk.CTkEntry(control_frame, textvariable=self.x1_var, width=200)
        self.x1_entry.grid(row=3, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Tolerancia
        ctk.CTkLabel(control_frame, text="Tolerancia (ε):", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        tol_entry = ctk.CTkEntry(control_frame, textvariable=self.tol_var, width=200)
        tol_entry.grid(row=4, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Máximo de iteraciones
        ctk.CTkLabel(control_frame, text="Máx. iteraciones:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        max_iter_entry = ctk.CTkEntry(control_frame, textvariable=self.max_iter_var, width=200)
        max_iter_entry.grid(row=5, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Criterio de parada
        ctk.CTkLabel(control_frame, text="Criterio de parada:", font=("Arial", 12, "bold")).grid(row=6, column=0, sticky="w", pady=5)
        criterio_cb = ctk.CTkComboBox(
            control_frame,
            variable=self.criterio_var,
            values=["error", "iteracion"],
            width=200
        )
        criterio_cb.grid(row=6, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Botones
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20, sticky="ew")
        
        run_button = ctk.CTkButton(
            button_frame,
            text="Ejecutar",
            command=self._run,
            width=100,
            height=40
        )
        run_button.pack(side="left", padx=5)
        
        clear_button = ctk.CTkButton(
            button_frame,
            text="Limpiar",
            command=self._clear,
            width=100,
            height=40
        )
        clear_button.pack(side="left", padx=5)
        
        # Área de salida
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        ctk.CTkLabel(output_frame, text="Salida LaTeX:", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Crear un textbox con scroll usando tkinter Text dentro de CTkScrollableFrame
        self.output = ctk.CTkTextbox(output_frame, height=300)
        self.output.pack(pady=5, padx=5, fill="both", expand=True)
        
        self._last_latex = ""

    def _on_method_change(self):
        method = self.method_var.get()
        if method == "secante":
            self.x1_entry.configure(state="normal")
        else:
            self.x1_entry.configure(state="disabled")

        if method == "punto_fijo":
            if self.func_var.get().strip() in ("", "x**2 - 2", "x**2-2"):
                self.func_var.set("0.5*(x + 2/x)")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
        else:
            if self.func_var.get().strip() in ("", "0.5*(x + 2/x)"):
                self.func_var.set("x**2 - 2")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
            if method == "secante" and self.x1_var.get().strip() == "":
                self.x1_var.set("2")

    def _clear(self):
        self.output.delete("0.0", "end")
        self._last_latex = ""

    def _run(self):
        func_str = self.func_var.get().strip()
        if not func_str:
            messagebox.showerror("Error", "La función no puede estar vacía")
            return

        try:
            x0 = float(self.x0_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(float(self.max_iter_var.get()))
            criterio = self.criterio_var.get().strip()

            if criterio not in ("error", "iteracion"):
                raise ValueError("Criterio debe ser 'error' o 'iteracion'")

            method = self.method_var.get()
            if method == "punto_fijo":
                iteraciones, latex = self.mn.punto_fijo(func_str, x0, tol, max_iter, criterio)
            elif method == "newton_raphson":
                iteraciones, latex = self.mn.newton_raphson(func_str, x0, tol, max_iter, criterio)
            elif method == "secante":
                x1 = float(self.x1_var.get())
                iteraciones, latex = self.mn.secante(func_str, x0, x1, tol, max_iter, criterio)
            else:
                raise ValueError("Método no soportado")

            self._last_latex = latex
            self.output.delete("1.0", "end")
            self.output.insert("1.0", latex)

            if not iteraciones:
                messagebox.showwarning("Atención",
                                       "No se generaron iteraciones (posible división por 0 o problema numérico).")

            self._save_pdf()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_pdf(self):
        if not self._last_latex:
            messagebox.showerror("Error", "No hay salida LaTeX. Primero ejecuta un método.")
            return

        if shutil.which("pdflatex") is None:
            messagebox.showerror(
                "Error",
                "No se encontró 'pdflatex'. Instala TeX Live/MiKTeX para generar PDF.",
            )
            return

        try:
            pdf_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("Todos", "*")],
                initialfile="resultado.pdf",
            )
            if not pdf_path:
                return

            with tempfile.TemporaryDirectory(prefix="mn_pdf_") as tmpdir:
                tex_filename = "resultado.tex"
                tex_path = os.path.join(tmpdir, tex_filename)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(self._last_latex)

                proc = None
                for _ in range(2):
                    proc = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        break

                tmp_pdf = os.path.join(tmpdir, "resultado.pdf")
                if proc is None or proc.returncode != 0 or not os.path.exists(tmp_pdf):
                    stdout = "" if proc is None else proc.stdout
                    stderr = "" if proc is None else proc.stderr
                    msg = stdout[-2000:] + "\n" + stderr[-2000:]
                    messagebox.showerror("Error", "Falló la compilación LaTeX.\n\n" + msg)
                    return

                shutil.copyfile(tmp_pdf, pdf_path)

                # Abrir el PDF automáticamente
                try:
                    webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
                except Exception as e:
                    print(f"No se pudo abrir el PDF automáticamente: {e}")

            messagebox.showinfo("OK", f"PDF generado en: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    app = MetodosNumericosGUI()
    app.root.mainloop()
