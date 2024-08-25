from system import clasificacion
import unittest

class test(unittest.TestCase):
    def test_clasificacion(self):
        sistema = clasificacion()
        resultados = sistema.modeloClasificacion()
        self.assertTrue(resultados["succes"])
        self.assertGreaterEqual(resultados["Precision"],0.7)
        if resultados["succes"]:
            print(f"El modelo se ejecutó correctamente. Precisión obtenida: {resultados['Precision']}")
        else:
            print(f"El modelo no se ejecutó correctamente.")
if __name__=="__main__":
    unittest.main()