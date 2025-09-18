# run.py
from app import app   # importa la instancia ya configurada
#print(type(app))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8055)