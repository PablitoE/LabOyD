from fpdf import FPDF


def balance_text(pdf: FPDF, text, width_col, **kwargs) -> tuple[str, str]:
    # 1. SIMULACIÓN: Obtenemos todas las líneas como si el texto fuera en una sola columna
    # split_only=True devuelve una lista de strings (cada línea renderizada)
    lineas_simuladas = pdf.multi_cell(w=width_col, txt=text, split_only=True, **kwargs)

    # 2. Calculamos el punto medio de las líneas
    total_lineas = len(lineas_simuladas)
    punto_medio = (total_lineas + 1) // 2  # El +1 es para que la izq sea >= der

    # 3. Agrupamos las líneas en dos bloques
    index = 0
    texto_izq = ""
    for i in range(punto_medio):
        texto_izq += lineas_simuladas[i]
        index += len(lineas_simuladas[i])
        next_start = text[index:].find(lineas_simuladas[i + 1])
        texto_izq += text[index:index + next_start]
        index += next_start
    texto_der = text[index:]
    return texto_izq, texto_der
