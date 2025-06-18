import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    x = 10
    return (x,)


@app.cell
def _(x):
    y = 2*x +1
    return (y,)


@app.cell
def _(y):
    print(f"{y=}")
    return


if __name__ == "__main__":
    app.run()
