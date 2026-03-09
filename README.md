# Software de Métodos Numéricos

Este programa implementa tres métodos numéricos para encontrar raíces de ecuaciones:

## Métodos Implementados

1. **Método de Aproximaciones Sucesivas (Punto Fijo)**
2. **Método de Newton-Raphson**
3. **Método de la Secante**

## Características

- **Entrada flexible**: Permite ingresar funciones como strings (ej: `x**2 - 2`, `cos(x) - x`, etc.)
- **Parámetros configurables**: 
  - Valor(es) inicial(es)
  - Tolerancia para criterio de convergencia
  - Máximo de iteraciones
  - Criterio de parada (por error o por número de iteraciones)
- **Salida en formato LaTeX**: Genera documentos LaTeX con tablas detalladas del proceso iterativo
- **Guardado automático**: Crea archivo `resultado.tex` con la salida LaTeX

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

Ejecutar el programa:

```bash
python main.py
```

Al ejecutarlo, se abrirá una interfaz gráfica (Tkinter) para seleccionar el método e ingresar los parámetros.

Si por alguna razón la interfaz gráfica no puede iniciarse (por ejemplo, en un entorno sin interfaz de ventanas), el programa cae automáticamente a un menú interactivo por consola.

## Ejemplos de Funciones

- `x**2 - 2` → Raíz: √2 ≈ 1.4142
- `x**3 - x - 1` → Raíz ≈ 1.3247
- `cos(x) - x` → Raíz ≈ 0.7391
- `exp(x) - 3*x` → Raíz ≈ 1.5121

## Formato de Salida LaTeX

El programa genera un documento LaTeX completo con:
- Encabezado con función y parámetros
- Tabla detallada del proceso iterativo
- Raíz aproximada y error final
- Formato matemático profesional

Para compilar el archivo `.tex` generado:
```bash
pdflatex resultado.tex
```
