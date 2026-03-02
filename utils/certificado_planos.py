import os

import pandas as pd

from src.Certificados.pdf_certificado import PDF
from src.Certificados.template_planos import FEIReader

nombre_archivo_excel = "resources/Certificados/template_planos.xlsx"
nombre_certificado = "Data/certificado.pdf"


if __name__ == "__main__":
    # Cargar el Excel
    df_especificaciones = pd.read_excel(nombre_archivo_excel, sheet_name="Especificaciones", header=None)
    df_especificaciones = df_especificaciones.dropna(how="all")  # Eliminar filas completamente vacías
    first_rows = ["TIPO TRABAJO", "Número", "Sub Trabajo", "Usuario", "Personal interviniente"]
    for i, expected_value in enumerate(first_rows):
        assert df_especificaciones.iloc[i, 0].strip().startswith(expected_value), \
            f"La fila {i+1} de la hoja 'Especificaciones' debe contener '{expected_value}' en la primera columna."
    assert df_especificaciones.iloc[len(first_rows), 0].strip().startswith('TIPO MUESTRA'), (
        'Luego de los datos deben haber elementos ingresados según plantilla.'
    )

    elements = []
    new_element = None
    for i in range(len(first_rows), len(df_especificaciones)):
        if df_especificaciones.iloc[i, 0].strip() == 'TIPO MUESTRA':
            if new_element is not None:
                elements.append(new_element)
            if df_especificaciones.iloc[i, 2] == 'No':
                new_element = None
            elif df_especificaciones.iloc[i, 2] == 'Si':
                if df_especificaciones.iloc[i, 1] in ('PLANO', 'JUEGO DE PARALELAS', 'PARALELA'):
                    new_element = {'tipo_muestra': df_especificaciones.iloc[i, 1]}
                else:
                    raise ValueError(f"Valor inesperado en 'TIPO MUESTRA': {df_especificaciones.iloc[i, 1]}")
            else:
                raise ValueError(f"Valor inesperado en USO de 'TIPO MUESTRA': {df_especificaciones.iloc[i, 2]}")
        elif new_element is not None:
            key = df_especificaciones.iloc[i, 0].strip().replace(' ', '_').replace(':', '')
            key = key.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            if key in new_element:
                raise ValueError(f"Clave duplicada '{key}' para el mismo tipo de muestra.")
            if df_especificaciones.iloc[i, 0].strip().startswith('Identificación usuario'):
                new_element[key] = df_especificaciones.iloc[i, 1:].dropna().tolist()
                for index, value in enumerate(new_element[key]):
                    if isinstance(value, float) and value.is_integer():
                        new_element[key][index] = int(value)
                new_element[key] = [str(x).strip() for x in new_element[key]]
                if len(new_element[key]) == 1:
                    new_element[key] = new_element[key][0]
            else:
                new_element[key] = df_especificaciones.iloc[i, 1]
    if new_element is not None:
        elements.append(new_element)

    # Distintos dataframes para las hojas
    df_paralelismo = pd.read_excel(nombre_archivo_excel, sheet_name="Paralelismo", header=None)

    df_incertidumbre_paralelismo = pd.read_excel(nombre_archivo_excel, sheet_name="Incert. paralelismo", header=None)

    df_planitud_tolerancia = pd.read_excel(nombre_archivo_excel, sheet_name="Planitud por Tolerancias", header=None)

    df_planitud_fei = pd.read_excel(nombre_archivo_excel, sheet_name="Planitud Flecha-Interfranja", header=None)
    df_planitud_fei = df_planitud_fei.dropna(how="all")  # Eliminar filas completamente vacías
    fei_data = FEIReader(df_planitud_fei)
    elements = fei_data.merge_with_elements(elements, "Identificacion_usuario")
    key_aliases = {
        "IMAGES_N": {"op": "entre_a",
                     "keys": ["planitud_fei.:.Cara_superior.n_ims",
                              "planitud_fei.:.Cara_inferior.n_ims"],
        }
    }

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
    datos_ot_rut = f"{df_especificaciones.iloc[0, 1]} {df_especificaciones.iloc[1, 1]} {df_especificaciones.iloc[2, 1]}"
    pdf = PDF(datos_ot_rut, font="stix", elements=elements, key_aliases=key_aliases)

    # PÁGINA 1: CARÁTULA
    pdf.set_caratula(df_caratula, width_label=50, vspace=10)

    # PÁGINA 2 de metodologia,Condiciones de medición y condiciones ambientales:
    # Para esta pagina hay que cambiar en el excel el valor del coeficiente de expansión
    pdf.add_sections(df_metodologia, vspace_after_text=0.5)

    # PÁGINA 3: RESULTADOS (tabla)
    pdf.add_sections(["# Resultados", "Los resultados se indican en la Tabla 1:"], vspace_after_text=3)
    # pdf.add_table_with_caption(df_para_word, "Tabla 1: Resultados de desviación al centro", extra_width=6)

    # PAGINA OBSERVACIONES
    pdf.add_sections(df_observaciones, vspace_after_text=3)

    # PÁGINA CIPM-MRA
    pdf.add_sections(df_mra, vspace_after_text=3, vspace_from_title=8, center_title=True, font_size_title=14)

    # Página de clausulas:
    pdf.add_sections(df_clausulas, vspace_after_text=20, vspace_from_title=8)

    # ----------------------------------------------------------------------------
    # Certificado:
    output_location = os.path.dirname(nombre_certificado)
    os.makedirs(output_location, exist_ok=True)
    pdf.output(nombre_certificado)
