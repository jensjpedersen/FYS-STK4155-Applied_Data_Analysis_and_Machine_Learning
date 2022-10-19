import poly_data
from importlib import reload

if __name__ == '__main__':
    reload(poly_data)
    p = poly_data.PolyData()
    p.plot()

