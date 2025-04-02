from flask import Flask, render_template, request, jsonify
from sympy import latex, symbols, Function, dsolve, integrate, Eq, sympify, exp, sin, cos, diff, Derivative
from sympy.abc import x
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import re

app = Flask(__name__)

# Función para convertir la entrada del usuario en una expresión SymPy válida
def parse_equation(eq):
    eq = re.sub(r"y''", "Derivative(y, x, x)", eq)
    eq = re.sub(r"y'", "Derivative(y, x)", eq)
    return eq

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_ode', methods=['POST'])
def solve_ode():
    data = request.json
    eq = data['equation']
    y = Function('y')(x) 

    try:
        if not re.search(r"y|y'|y''", eq):
            return jsonify({'error': 'Lo siento, escribe correctamente una ecuación diferencial.'})

        eq_parsed = parse_equation(eq)

        if '=' in eq_parsed:
            left_side, right_side = eq_parsed.split('=')
            left_expr = sympify(left_side.strip(), locals={'y': y, 'x': x, 'Derivative': Derivative})
            right_expr = sympify(right_side.strip(), locals={'y': y, 'x': x, 'Derivative': Derivative})
            equation = Eq(left_expr, right_expr)
        else:
            expr = sympify(eq_parsed, locals={'y': y, 'x': x, 'Derivative': Derivative})
            equation = Eq(expr, 0)

        sol = dsolve(equation, y)
        return jsonify({'solution': latex(sol)})
    except (SyntaxError, ValueError):
        return jsonify({'error': 'Lo siento, escribe correctamente la ecuación diferencial.'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Función para graficar la integral
def plot_integral_func(expr_func, lower, upper):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    x_vals = np.linspace(lower, upper, 100)
    y_vals = [float(integrate(expr_func, (x, lower, val))) for val in x_vals]

    ax.plot(x_vals, y_vals)
    ax.grid(True)
    ax.set_title(f'Gráfico de la integral de {latex(expr_func)} de {lower} a {upper}')
    ax.set_xlabel('x')
    ax.set_ylabel('Integral')

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close(fig)

    return plot_url

@app.route('/integrate_indefinite', methods=['POST'])
def integrate_indefinite():
    data = request.json
    expr = data['expression']
    try:
        # Convertir la expresión directamente usando los imports existentes
        expr_sympy = sympify(expr)
        
        # Calcular la integral
        result = integrate(expr_sympy, x)
        
        # Ajustar límites del gráfico para evitar singularidades
        lower = 0.1 if '/x' in expr.replace(' ', '') else -5
        upper = 5
        
        plot_url = plot_integral_func(expr_sympy, lower, upper)
        
        return jsonify({
            'result': latex(result),
            'plot_url': plot_url
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error: Usa sintaxis como sqrt(x) + 1/x o x*e^(x^2)'
        })
    
@app.route('/integrate_definite', methods=['POST'])
def integrate_definite():
    data = request.json
    expr = data['expression']
    lower = data['lower']
    upper = data['upper']
    try:
        result = integrate(sympify(expr), (x, lower, upper))
        plot_url = plot_integral_func(sympify(expr), lower, upper)
        return jsonify({'result': latex(result), 'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': 'Lo siento, escribe correctamente una expresión matemática'})

@app.route('/plot_integral', methods=['POST'])
def plot_integral():
    data = request.json
    expr = data['expression']
    lower = float(data['lower'])
    upper = float(data['upper'])
    try:
        # Primero calculamos el resultado de la integral definida
        expr_func = sympify(expr)
        definite_result = integrate(expr_func, (x, lower, upper))
        
        # Luego generamos el gráfico
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        x_vals = np.linspace(lower, upper, 100)
        y_vals = [float(integrate(expr_func, (x, lower, val))) for val in x_vals]
        
        ax.plot(x_vals, y_vals)
        ax.grid(True)
        ax.set_title(f'Gráfico de la integral de {latex(expr_func)} de {lower} a {upper}')
        ax.set_xlabel('x')
        ax.set_ylabel('Integral')

        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)
        
        return jsonify({
            'result': latex(definite_result),
            'plot_url': plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

from prediction import prediction_bp
app.register_blueprint(prediction_bp, url_prefix='/prediction_api')

if __name__ == '__main__':
    app.run(debug=True)
