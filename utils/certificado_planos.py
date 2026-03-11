import os

import pandas as pd

import src.Certificados.template_planos as tp
from src.Certificados.pdf_certificado import PDF

nombre_archivo_excel = "resources/Certificados/template_planos.xlsx"
nombre_certificado = "Data/certificado.pdf"


if __name__ == "__main__":
    # Cargar el Excel
    df_especificaciones = pd.read_excel(nombre_archivo_excel, sheet_name="Especificaciones", header=None)
    df_especificaciones = df_especificaciones.dropna(how="all")  # Eliminar filas completamente vacías
    specs_data = tp.SpecsReader(df_especificaciones)
    elements = specs_data.get_elements()

    # Distintos dataframes para las hojas
    df_paralelismo = pd.read_excel(nombre_archivo_excel, sheet_name="Paralelismo", header=None)
    df_incertidumbre_paralelismo = pd.read_excel(nombre_archivo_excel, sheet_name="Incert. paralelismo", header=None)
    parallelism_data = tp.ParallelismReader(df_paralelismo, df_incertidumbre_paralelismo)

    df_planitud_tolerancia = pd.read_excel(nombre_archivo_excel, sheet_name="Planitud por Tolerancias", header=None)
    tolerancia_data = tp.ToleranceReader(df_planitud_tolerancia)
    subfigs = tolerancia_data.prepare_subfigs(elements)

    df_planitud_fei = pd.read_excel(nombre_archivo_excel, sheet_name="Planitud Flecha-Interfranja", header=None)
    df_planitud_fei = df_planitud_fei.dropna(how="all")  # Eliminar filas completamente vacías
    fei_data = tp.FEIReader(df_planitud_fei)
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

    # Dataframe secciones de Resultados
    df_resultados = pd.read_excel(nombre_archivo_excel, sheet_name="RESULTADOS", header=None)

    # ----------------------------------------------------------------------------
    # Crear instancia del PDF
    datos_ot_rut = f"{df_especificaciones.iloc[0, 1]} {df_especificaciones.iloc[1, 1]} {df_especificaciones.iloc[2, 1]}"
    pdf = PDF(datos_ot_rut, font="Arial", elements=elements, key_aliases=key_aliases)

    # PÁGINA 1: CARÁTULA
    pdf.set_caratula(df_caratula, width_label=50, vspace=10)

    # PÁGINA 2 de metodologia,Condiciones de medición y condiciones ambientales:
    # Para esta pagina hay que cambiar en el excel el valor del coeficiente de expansión
    pdf.add_sections(df_metodologia, vspace_after_text=0.5)

    df_table_fei = fei_data.build_table(elements)
    df_table_parallelism = parallelism_data.build_table(elements)
    pdf.add_sections(df_resultados, vspace_after_text=3, table_dfs=[df_table_fei, df_table_parallelism],
                     subfigs_dfs=subfigs, in_new_page=False)

    # PAGINA OBSERVACIONES
    pdf.add_sections(df_observaciones, vspace_after_text=3, in_new_page=False)

    # PÁGINA CIPM-MRA
    pdf.add_sections(df_mra, vspace_after_text=3, vspace_from_title=8, center_title=True, font_size_title=14)

    # Página de clausulas:
    pdf.add_sections(df_clausulas, vspace_after_text=20, vspace_from_title=8)

    # ----------------------------------------------------------------------------
    # Certificado:
    output_location = os.path.dirname(nombre_certificado)
    os.makedirs(output_location, exist_ok=True)
    pdf.output(nombre_certificado)
