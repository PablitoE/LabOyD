import pandas as pd

from src.Certificados.pdf_certificado import PDF

nombre_archivo_excel = "resources/Certificados/template_bp.xlsx"
DATOS_OT_RUT = "RUT N° 7100470421 Único"
nombre_certificado = "122BP.pdf"


if __name__ == "__main__":
    # 1. Cargar el Excel
    df = pd.read_excel(nombre_archivo_excel, sheet_name="Resultados")
    # --------------------------------------------------------------------------
    df = df.dropna(how="all")

    # 3. FILTRAR POR UNA COLUMNA CLAVE
    # Si la columna 'N' es la que tiene el número de medición,
    # nos aseguramos de que solo queden filas donde 'N' tenga un valor real.
    df = df[df["N"].notna()]
    # -----------------------------------------------------------------------------
    # Elegir las columnas que quiero (Datos para la tabla)
    # Simplemente pasamos una lista con los nombres exactos de las columnas
    columnas_que_quiero = [
        "N",
        "L(mm)",
        "Identificacion",
        "fn (nm)",
        "cara adherida",
        "U (nm)",
    ]
    df_para_word = df[
        columnas_que_quiero
    ].copy()  # Dataframe son las columnas que me intersan para el certificado

    # Hay columnas que son numeros y otras no
    df_para_word["Identificacion"] = df_para_word["Identificacion"].map("{:.0f}".format)
    df_para_word["N"] = df_para_word["N"].astype(int)
    # Esto los convierte en texto con el formato de cifras que quiero ver
    df_para_word["L(mm)"] = df_para_word["L(mm)"].astype(str)
    df_para_word["fn (nm)"] = df_para_word["fn (nm)"].map("{:.0f}".format)
    df_para_word["U (nm)"] = df_para_word["U (nm)"].map("{:.0f}".format)
    # ------------------------------------------------------------------------------
    # Distintos dataframes para las hojas
    # DataFrame con cosas para la caratula:
    df_caratula = pd.read_excel(nombre_archivo_excel, sheet_name="CARATULA", header=None)

    # Dataframe texto fijo de la anteultima pagina (CIPM MRA):
    df_mra = pd.read_excel(nombre_archivo_excel, sheet_name="MRA", header=None)

    # Dataframe Texto fijo ultima página clausulas del certificado:
    df_clausulas = pd.read_excel(nombre_archivo_excel, sheet_name="CLAUSULAS", header=None)

    # dataframe Texto pagina parrafo de incertidumbre & observaciones:
    df_observaciones = pd.read_excel(nombre_archivo_excel, sheet_name='OBSERVACIONES', header=None)

    # Dataframe secciones Metodología empleada,Condiciones de medición y Condiciones ambientales
    df_metodologia = pd.read_excel(nombre_archivo_excel, sheet_name="METODOLOGIA", header=None)

    # ----------------------------------------------------------------------------
    # Crear instancia del PDF
    pdf = PDF(DATOS_OT_RUT)

    # PÁGINA 1: CARÁTULA
    pdf.set_caratula(df_caratula, width_label=50, vspace=10)

    # PÁGINA 2 de metodologia,Condiciones de medición y condiciones ambientales:
    # Para esta pagina hay que cambiar en el excel el valor del coeficiente de expansión
    pdf.add_sections(df_metodologia, vspace_after_text=3)

    # PÁGINA 3: RESULTADOS (tabla)
    pdf.add_sections(["# Resultados", "Los resultados se indican en la Tabla 1:"], vspace_after_text=3)
    pdf.add_table_with_caption(df_para_word, "Tabla 1: Resultados de desviación al centro", extra_width=6)

    # PAGINA OBSERVACIONES
    pdf.add_sections(df_observaciones, vspace_after_text=3)

    # PÁGINA CIPM-MRA
    pdf.add_sections(df_mra, vspace_after_text=3, vspace_from_title=8, center_title=True, font_size_title=14)

    # Página de clausulas:
    pdf.add_sections(df_clausulas, vspace_after_text=20, vspace_from_title=8)

    # ----------------------------------------------------------------------------
    # Certificado:
    pdf.output(nombre_certificado)
