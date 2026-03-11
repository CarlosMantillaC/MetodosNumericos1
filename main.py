#!/usr/bin/env python3
"""
Métodos Numéricos - Aplicación para encontrar raíces de ecuaciones

Esta aplicación implementa varios métodos numéricos para encontrar raíces de ecuaciones:
- Método de Aproximaciones Sucesivas (Punto Fijo)
- Método de Newton-Raphson
- Método de la Secante

La aplicación genera informes en formato LaTeX con los resultados de cada método.
"""

from gui import NumericalMethodsGUI


def main():
    """Función principal que inicia la aplicación."""
    app = NumericalMethodsGUI()
    app.root.mainloop()


if __name__ == '__main__':
    main()
