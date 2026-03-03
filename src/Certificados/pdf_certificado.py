import logging
import os

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pandas import DataFrame

from src.Certificados.pdf_extras import balance_text, estimate_lines

logger = logging.getLogger(__name__)


class PDF(FPDF):
    def __init__(self, datos_ot_rut, resource_path="resources/Certificados", font="helvetica", elements=None,
                 key_aliases=None):
        super().__init__()
        self.counter_table = 0
        self.tables = []
        self.add_font("stix", "", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Regular.otf"), uni=True)
        self.add_font("stix", "B", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Bold.otf"), uni=True)
        self.add_font("stix", "I", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Italic.otf"), uni=True)
        self.add_font("stix", "BI", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-BoldItalic.otf"),
                      uni=True)

        self.datos_ot_rut = datos_ot_rut
        self.resource_path = resource_path
        self.font = font
        self.alias_nb_pages()  # Funcion que hay que llamar para poder numerar las paginas tipon "1 de"
        self.set_auto_page_break(auto=True, margin=30)
        self.set_top_margin(55)

        self.buffer_two_columns = ""  # Para almacenar temporalmente el texto de dos columnas
        self.elements = elements  # Para almacenar información de elementos como tipo de muestra, identificación, etc.
        self.elements_info = {}  # Para almacenar información procesada de elementos, evita procesamientos repetidos
        self.key_aliases = key_aliases if key_aliases is not None else {}  # Para mapear claves más legibles en el texto

    def header(self):  # Encabezado
        # 1. Imagen de encabezado
        # Mantenemos la imagen donde estaba
        self.image(os.path.join(self.resource_path, "encabezado1.jpg"), 10, 8, 199)

        # 2. RUT (Lo subimos y achicamos su celda)
        # Ponemos el cursor a 25mm del borde superior
        self.set_y(30)
        self.set_font(self.font, "", 9)
        self.cell(0, 4, self.datos_ot_rut, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        # 3. Título del Certificado
        self.set_font(self.font, "B", 18)
        self.cell(0, 10, 'Certificado de Calibración/Medición', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='l')
        self.set_font(self.font, "", 9)
        self.set_text_color(0)  # Negro
        texto_paginas = f"Página {self.page_no()} de {{nb}}"
        self.cell(0, 10, texto_paginas, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        # 4. Espacio final antes del contenido
        self.ln(5)

    def footer(self):  # Pie
        # 1. Posicionamos a -20 mm del final de la página
        # (Ajusta este número si la imagen es muy alta)
        self.set_y(-25)
        # 2. Insertamos la imagen
        # x=10 (margen izquierdo), ancho=190 (casi todo el ancho A4)
        self.image(os.path.join(self.resource_path, "pie1.jpg"), x=10, y=self.get_y(), w=190)

    @property
    def effective_page_width(self):
        return self.w - self.l_margin - self.r_margin

    def set_caratula(self, df: DataFrame, width_label=50, vspace=10) -> None:
        self.add_page()

        row_label = 0
        while row_label < len(df):
            label = str(df.iloc[row_label, 0])
            data = []
            if not pd.isna(df.iloc[row_label, 0]):
                while pd.isna(df.iloc[row_label, 1]):
                    row_label += 1
                    if row_label == len(df):
                        break
                    label += " " + str(df.iloc[row_label, 0])
                else:
                    data = [df.iloc[row_label, 1]]
                    row_data = row_label + 1
                    while row_data < len(df) and pd.isna(df.iloc[row_data, 0]) and not pd.isna(df.iloc[row_data, 1]):
                        data.append(df.iloc[row_data, 1])
                        row_data += 1
                    value = "\n".join(map(str, data))
                    self.set_font(self.font, "B", 10)
                    self.multi_cell(width_label, 5, label, border=0, new_x=XPos.RIGHT, new_y=YPos.LAST, align="l")
                    self.set_font(self.font, "", 10)
                    self.multi_cell(0, 5, value, border=0, align="L")
                row_label += len(data)
                self.ln(vspace)
            else:
                if not pd.isna(df.iloc[row_label, 1]):
                    logger.warning(
                        f"Fila {row_label} tiene etiqueta vacía pero valor no vacío: '{df.iloc[row_label, 1]}'"
                    )
                row_label += 1

    def add_sections(self, df: DataFrame, vspace_before_title=2, vspace_from_title=2, vspace_after_text=0,
                     center_title=False, font_size=10, font_size_title=12, font_size_two_columns=8.5,
                     in_new_page=True, table_dfs=None) -> None:

        """
        Agrega secciones a un PDF desde un DataFrame.

        Modos de interpretación del DataFrame:
        Título: La primera columna del DataFrame debe contener el título de la sección, comenzando con "#".
                Por ejemplo, "# Metodología Empleada".
        Texto normal: Sólo primera columna con texto, sin formato especial. Si el texto comienza con "*", se interpreta
                      como texto en negrita.
        Texto en dos columnas: Si la segunda columna tiene el valor "two_columns", el texto de la primera columna se
                              acumula hasta que se encuentra una fila que no tiene "two_columns" en la segunda columna.
        Imágenes: Si la primera columna comienza con "_fig" y la segunda columna contiene el nombre de un archivo de
                  imagen, se interpreta como una figura a insertar. El caption de la figura se toma del texto que sigue
                  a "_fig" en la primera columna, y el ancho de la imagen se toma de la tercera columna.


        Parameters
        ----------
        df : DataFrame
        vspace_from_title : int, optional (mm, [2])
        vspace_after_text : int, optional (mm, [0])
        center_title : bool, optional [False]
        font_size : int, optional [10]
        font_size_title : int, optional [12]
        font_size_two_columns : int, optional [8.5]
        in_new_page : bool, optional [True]
        """
        if in_new_page:
            self.add_page()
        if isinstance(df, list):
            df = pd.DataFrame(df)
        assert str(df.iloc[0, 0])[0] == "#", "La primera fila debe contener el título de la sección, comenzando con '#'"

        row = 0
        while row < len(df):
            if str(df.iloc[row, 0])[0:2] == "# ":
                self.set_font(self.font, "B", size=font_size_title)
                self.ln(vspace_before_title)
                title = str(df.iloc[row, 0])[2:]
                alignment = "C" if center_title else "L"
                self.cell(0, 5, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=alignment)
                self.ln(vspace_from_title)
            elif str(df.iloc[row, 0])[0:4] == "_fig" and pd.notna(df.iloc[row, 1]):
                caption = str(df.iloc[row, 0])[4:].strip()
                file_path = os.path.join(self.resource_path, str(df.iloc[row, 1]))
                width_mm = df.iloc[row, 2] if df.shape[1] > 2 and pd.notna(df.iloc[row, 2]) else None
                self.add_figure(file_path, caption=caption, width_mm=width_mm)
            elif str(df.iloc[row, 0])[0:4] == "_tab":
                sep_index = str(df.iloc[row, 0]).find(" ")
                ref = str(df.iloc[row, 0])[:sep_index]
                caption = str(df.iloc[row, 0])[sep_index + 1:].strip()
                assert table_dfs is not None, "Se encontró una fila de tabla pero no se proporcionó 'table_dfs'"
                if isinstance(table_dfs, list):
                    assert len(table_dfs) > self.counter_table, "No hay suficientes DataFrames para las tablas"
                    df_table = table_dfs[self.counter_table]
                else:
                    df_table = table_dfs
                self.counter_table += 1
                self.tables.append({"num": self.counter_table, "ref": ref})
                self.add_table_with_caption(df_table, caption, contains_multirows=True, extra_width=2, row_height=12)
            elif not pd.isna(df.iloc[row, 0]):
                self.set_font(self.font, "", size=font_size)
                writing = True
                if df.shape[1] == 1 or pd.isna(df.iloc[row, 1]):
                    text = str(df.iloc[row, 0])
                    if text.startswith("*"):
                        text = text[1:].strip()
                        self.set_font(style="B")
                    self.multi_cell(0, 5, text, border=0, align="J")
                elif df.shape[1] > 1 and df.iloc[row, 1] == "two_columns":
                    self.buffer_two_columns += str(df.iloc[row, 0]) + "\n"
                    if row + 1 >= len(df) or df.iloc[row + 1, 1] != "two_columns":
                        self.add_balanced_text(font_size=font_size_two_columns, h=5, align="J")
                        self.buffer_two_columns = ""
                    else:
                        writing = False
                elif df.shape[1] > 1 and df.iloc[row, 1] == "**" and self.elements is not None:
                    key, value = str(df.iloc[row, 2]).split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in self.elements_info:
                        self.process_elements(key)
                    if value in self.elements_info[key]:
                        text = str(df.iloc[row, 0])
                        self.multi_cell(0, 5, text, border=0, align="J")
                elif df.shape[1] > 1 and df.iloc[row, 1] == "$$" and self.elements is not None:
                    key, value = str(df.iloc[row, 2]).split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    base_text = str(df.iloc[row, 0])
                    for element in self.elements:
                        if key in element and value in element[key]:
                            text = self.insert_values_into_text(base_text, element)
                            self.multi_cell(0, 5, text, border=0, align="J")
                            break
                elif not pd.isna(df.iloc[row, 1]):
                    self.write_value_with_units(df.iloc[row, 0], df.iloc[row, 1])
                else:
                    raise ValueError(f"Fila {row} tiene formato inesperado: '{df.iloc[row, 0]}', '{df.iloc[row, 1]}'")
                if vspace_after_text > 0 and writing:
                    self.ln(vspace_after_text)
            row += 1

    def write_value_with_units(self, value, html_units):
        expansion = f"{value} {html_units}"
        self.write_html(expansion)
        self.ln(10)  # write_html no da salto de línea automático al final, hay que agregarlo

    def add_table_with_caption(
        self, df: DataFrame, caption: str, caption_height=10, vspace_after_table=5, fit_col_widths=True,
        padding_lr=2, padding_tb=0, extra_width=0, row_height=5, contains_multirows=False
    ):
        """
        Agrega una tabla a un PDF con un caption

        Si una celda contiene un texto que termina con ">" se interpreta como una celda que abarca tantas columnas
        adicionales como ">" tenga.
        """
        self.set_font(self.font, "", size=10)
        caption = f"Tabla {self.counter_table}: {caption}"
        self.cell(0, caption_height, caption, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        if fit_col_widths:
            col_widths = []
            for col in df.columns:
                max_content_width = self.get_string_width(str(col))
                for val in df[col]:
                    val_width = self.get_string_width(str(val))
                    if val_width > max_content_width:
                        max_content_width = val_width
                col_widths.append(max_content_width + 2 * padding_lr)  # Agregar un poco de padding
            sum_col_widths = sum(col_widths)
            proportional_widths = [width / sum_col_widths for width in col_widths]
            total_width = sum_col_widths + extra_width * len(df.columns)  # Agregar espacio para los bordes
            total_width = self.effective_page_width if total_width > self.effective_page_width else total_width
            col_widths = [width_ratio * total_width for width_ratio in proportional_widths]
        else:
            col_widths = None
            total_width = self.effective_page_width

        if contains_multirows:
            start_x = self.w / 2 - total_width / 2
            self.set_xy(start_x, self.get_y())
            self.set_font(style="B")
            n_ln = 1
            for col_k, col_name in enumerate(df.columns):
                this_n_ln, _ = estimate_lines(self, col_name, col_widths[col_k])
                n_ln = max(n_ln, this_n_ln)
            for col_k, col_name in enumerate(df.columns):
                self.multi_cell(col_widths[col_k], row_height * n_ln, col_name, border=1, align="C",
                                new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size,
                                padding=(padding_tb, padding_lr))
            self.set_xy(start_x, self.get_y() + row_height * n_ln)
            self.set_font(style="")
            n_span_below = [0] * len(df.columns)
            for _, fila in df.iterrows():
                n_span_right = 0
                for col_k, val in enumerate(fila):
                    do_continue = False
                    if n_span_right > 0:
                        n_span_right -= 1
                        do_continue = True
                    if n_span_below[col_k] > 0:
                        n_span_below[col_k] -= 1
                        do_continue = True
                    if do_continue:
                        self.set_x(self.get_x() + col_widths[col_k])
                        continue
                    val_str = str(val)
                    n_span_right = len(val_str) - len(val_str.rstrip(">"))
                    n_span_below[col_k] = len(val_str) - len(val_str.lstrip("_"))
                    val_str = val_str.rstrip(">").lstrip("_")
                    self.multi_cell(col_widths[col_k] * (n_span_right + 1), row_height * (n_span_below[col_k] + 1),
                                    val_str, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP,
                                    max_line_height=self.font_size, padding=(padding_tb, padding_lr))
                self.set_xy(start_x, self.get_y() + row_height)
        else:
            with self.table(
                width=total_width, first_row_as_headings=True, text_align='C', line_height=row_height, align='C',
                padding=(padding_tb, padding_lr), col_widths=col_widths
            ) as table:
                self.set_font(style="B")
                row = table.row()
                for col_name in df.columns:
                    row.cell(col_name)

                self.set_font(style="")
                for _, fila in df.iterrows():
                    row = table.row()
                    n_span = 0
                    for valor in fila:
                        if n_span > 0:
                            n_span -= 1
                            continue
                        valor = str(valor)
                        n_span = len(valor) - len(valor.rstrip(">"))
                        row.cell(valor.rstrip(">"), colspan=n_span + 1)
        self.ln(vspace_after_table)

    def add_figure(self, filename, caption=None, width_mm=None, vspace_before_caption=2, vspace_after_figure=3):
        if width_mm is None:
            width_mm = self.effective_page_width * 0.8  # Por defecto, ocupar el 80% del ancho efectivo
        x_center = self.l_margin + (self.effective_page_width - width_mm) / 2
        self.image(filename, x=x_center, w=width_mm)

        if caption:
            self.set_y(self.get_y() + vspace_before_caption)
            self.set_font(self.font, style="I", size=9)
            self.cell(w=self.effective_page_width, h=5, txt=caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(vspace_after_figure)

    def add_balanced_text(self, width_col=None, gutter=6, font_size=10, **kwargs):
        font_size_start = self.font_size_pt
        self.set_font(size=font_size)
        if width_col is None:
            width_col = (self.effective_page_width - gutter) / 2
        texto_izq, texto_der = balance_text(self, self.buffer_two_columns, width_col=width_col, **kwargs)
        x_col2 = self.l_margin + width_col + gutter  # Posición X para la segunda columna
        y_inicio = self.get_y()

        # Columna izquierda
        self.set_x(self.l_margin)
        self.multi_cell(width_col, txt=texto_izq, border=0, **kwargs)

        # Columna derecha
        y_final_col1 = self.get_y()
        self.set_xy(x_col2, y_inicio)
        self.multi_cell(width_col, txt=texto_der, border=0, **kwargs)

        # Ajustar Y para que quede al final del bloque más alto
        y_final = max(y_final_col1, self.get_y())
        self.set_y(y_final)
        self.set_font(size=font_size_start)

    def process_elements(self, key):
        if key in self.elements_info:
            return  # Ya procesamos este elemento antes, no es necesario hacerlo de nuevo
        self.elements_info[key] = []
        for element in self.elements:
            value = element.get(key)
            self.elements_info[key].append(value)
        if not self.elements_info[key]:
            self.elements_info[key] = None

    def insert_values_into_text(self, base_text, list_of_dicts, sep="$"):
        if isinstance(list_of_dicts, dict):
            list_of_dicts = [list_of_dicts]
        start_variable = 1 if base_text[0] != "$" else 0
        text = base_text.split(sep)
        for i in range(start_variable, len(text), 2):
            for d in list_of_dicts:
                if text[i] in self.key_aliases:
                    actual_key = self.key_aliases[text[i]]
                    value = self.deep_get(d, actual_key)
                elif text[i] in d and d[text[i]] is not None:
                    value = d[text[i]]
                else:
                    value = None
                if value is not None:
                    text[i] = str(value)
                    break
        return "".join(text)

    @staticmethod
    def deep_get(data, alias, default=None, sep="."):
        """
        Accede a una estructura anidada usando un string tipo "a.b.0.c".
        Parameters
        ----------
        data : dict | list
            Estructura donde buscar.
        alias : str | list | dict
            Ruta separada por `sep`.
        default : any
            Valor a devolver si alguna clave no existe.
        sep : str
            Separador de niveles (default ".").

        Returns
        -------
        any
            Valor encontrado o `default` si falla.
        """
        if isinstance(alias, str):
            alias = {"op": None, "keys": alias}
        if isinstance(alias["keys"], str):
            alias["keys"] = [alias["keys"]]
        output = []
        for key_for_alias in alias["keys"]:
            keys = key_for_alias.split(sep)
            value = data
            flag_all = False
            for k in keys:
                try:
                    if k == ":":
                        flag_all = isinstance(value, list)
                        continue
                    if isinstance(value, list):
                        if flag_all:
                            new_value = []
                            for v in value:
                                if v is not None and k in v:
                                    new_value.append(v[k])
                                else:
                                    new_value.append(default)
                            value = new_value
                            continue
                        k = int(k)  # acceso por índice
                    value = value[k]
                except (KeyError, IndexError, ValueError, TypeError):
                    value = default
            output.append(value)
        if len(alias["keys"]) == 1:
            return output[0]
        if alias["op"] == "entre_a":
            minimum = float("inf")
            maximum = -float("inf")
            for values in output:
                if values is not None and isinstance(values, list):
                    values = [v for v in values if v is not None]
                    if not values:
                        continue
                    val_min = min(values)
                    val_max = max(values)
                    minimum = val_min if val_min < minimum else minimum
                    maximum = val_max if val_max > maximum else maximum
                elif values is not None:
                    minimum = values if values < minimum else minimum
                    maximum = values if values > maximum else maximum
            if minimum < maximum:
                return f"entre {minimum} a {maximum}"
            else:
                return f"{minimum}"

        return output


if __name__ == "__main__":
    print(
        "Este módulo define la clase PDF para generar certificados.",
         " No se ejecuta nada al correr este archivo directamente."
    )
