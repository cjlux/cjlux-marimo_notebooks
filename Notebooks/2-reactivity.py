import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Marimo is REACTIVE""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$\leadsto$ when you modify an object definition, all the cells using this object are _automatically_ run:""")
    return


@app.cell
def _():
    a = 180               # angle in degrees
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The next cell uses $x$ in a f_string:""")
    return


@app.cell
def _(a):
    print(f'the value of a is: {a} [°]')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next cell uses $a$ to define $y$:""")
    return


@app.cell
def _(a, np):
    y = np.cos(np.radians(a))
    float(y)
    return (y,)


@app.cell(hide_code=True)
def _(a, mo, y):
    mo.md(rf"""You can even use _f_strings_ inside a mardown cell like this one: rounded value of cos({a}°) $\leadsto$ {y:.3f}""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


if __name__ == "__main__":
    app.run()
