import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interactive widgets (UI elements)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Slider""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Simple usage""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Variation using the *cell output* (visual output)""")
    return


@app.cell
def _(mo):
    s = mo.ui.slider(1,10)
    s
    return (s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**print(...)** displays data using the *console output* (display updates flash a bit)""")
    return


@app.cell
def _(s):
    print(f"value of the 's' slider: {s.value}")
    return


@app.cell(hide_code=True)
def _(mo):
    s2 = mo.ui.slider(1,10,2)
    mo.md(f"Choose a value : {s2}")
    return (s2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | **cell last line** displays data using the *cell output* 

    Using *markdown* in the last line gives smoother display update

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, s2):
    mo.md(f"""slider 's2' value: {s2.value}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Date""")
    return


@app.cell
def _(mo):
    dates = mo.md("{start} → {end}").batch(
        start=mo.ui.date(label="Start Date"),
        end=mo.ui.date(label="End Date")
    )
    dates
    return (dates,)


@app.cell
def _(dates):
    dates.value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Other marimo input here : https://docs.marimo.io/api/inputs/""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib.pyplot import plot
    return (mo,)


if __name__ == "__main__":
    app.run()
