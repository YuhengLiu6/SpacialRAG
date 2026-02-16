import habitat_sim.utils.datasets_download as dd
import inspect

try:
    print(inspect.getsource(dd))
except Exception as e:
    print(f"Error reading source: {e}")
    print(f"Module file: {dd.__file__}")
