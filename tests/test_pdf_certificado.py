from Certificados.pdf_certificado import PDF


def test_pdf_certificado(in_main=False):
    pdf = PDF("")
    partes, tipos = pdf._parse_seps(
        "Esto es una prueba de separadores *primero*, $segundo$ y /tercero/. Este va combinado: /cuarto y $quinto$/",
        ("*", "$", "/")
    )
    if in_main:
        print(partes)
        print(tipos)
    else:
        assert len(partes) == 9
        assert tipos == ["text", "*", "text", "$", "text", "/", "text", "/", "/$"]


if __name__ == "__main__":
    test_pdf_certificado(in_main=True)
