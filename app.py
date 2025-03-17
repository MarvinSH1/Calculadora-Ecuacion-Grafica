from flask import Flask, render_template, request, jsonify
from sympy import symbols, Function, dsolve, integrate, Eq, plot, sympify
from sympy.abc import x
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_ode', methods=['POST'])
def solve_ode():
    data = request.json
    eq = data['equation']
    y = Function('y')(x)  # Definir y como función de x
    try:
        # Convertir la ecuación ingresada a una expresión SymPy
        expr = sympify(eq, locals={'y': y, 'x': x, 'diff': y.diff(x)})
        # Resolver la ecuación diferencial
        sol = dsolve(Eq(expr, 0), y)
        return jsonify({'solution': str(sol)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/integrate_indefinite', methods=['POST'])
def integrate_indefinite():
    data = request.json
    expr = data['expression']
    try:
        result = integrate(eval(expr), x)
        return jsonify({'result': str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/integrate_definite', methods=['POST'])
def integrate_definite():
    data = request.json
    expr = data['expression']
    lower = data['lower']
    upper = data['upper']
    try:
        result = integrate(eval(expr), (x, lower, upper))
        return jsonify({'result': str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot_integral', methods=['POST'])
def plot_integral():
    data = request.json
    expr = data['expression']
    lower = data['lower']
    upper = data['upper']
    try:
        p = plot(integrate(eval(expr), (x, lower, upper)), show=False)
        img = io.BytesIO()
        p.save(img)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return jsonify({'plot_url': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
