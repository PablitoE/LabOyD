import logging
import os

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.image_parsing import get_img_info
from pandas import DataFrame

import Certificados.pdf_extras as extras

logger = logging.getLogger(__name__)


class PDF(FPDF):
    def __init__(self, datos_ot_rut, resource_path="resources/Certificados", font="helvetica", elements=None,
                 key_aliases=None, general_info=None):
        super().__init__()
        self.add_font("stix", "", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Regular.otf"), uni=True)
        self.add_font("stix", "B", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Bold.otf"), uni=True)
        self.add_font("stix", "I", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-Italic.otf"), uni=True)
        self.add_font("stix", "BI", os.path.join(resource_path, "fonts", "stix", "STIXGeneral-BoldItalic.otf"),
                      uni=True)
        self.add_font("Arial", "", os.path.join(resource_path, "fonts", "arial", "Arial.ttf"), uni=True)
        self.add_font("Arial", "B", os.path.join(resource_path, "fonts", "arial", "Arial_Bold.ttf"), uni=True)
        self.add_font("Arial", "I", os.path.join(resource_path, "fonts", "arial", "Arial_Italic.ttf"), uni=True)
        self.add_font("Arial", "BI", os.path.join(resource_path, "fonts", "arial", "Arial_Bold_Italic.ttf"),
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
        self.general_info = general_info
        self.key_aliases = key_aliases if key_aliases is not None else {}  # Para mapear claves más legibles en el texto
        self.current_iteration_total = None
        self.current_iteration_element = None

        if self.general_info is None:
            self.general_info = {}
        self.counter_table = 0
        self.tables = []
        self.counter_figure = 0
        self.figures = []

    @property
    def counter_table(self):
        return self._counter_table

    @counter_table.setter
    def counter_table(self, value):
        self._counter_table = value
        self.general_info["tab_this"] = value
        self.general_info["tab_next"] = value + 1
        self.general_info["tab_prev"] = value - 1

    @property
    def counter_figure(self):
        return self._counter_figure

    @counter_figure.setter
    def counter_figure(self, value):
        self._counter_figure = value
        self.general_info["fig_this"] = value
        self.general_info["fig_next"] = value + 1
        self.general_info["fig_prev"] = value - 1


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
                        value = "" if "\\" == str(df.iloc[row_data, 1]) else df.iloc[row_data, 1]
                        data.append(value)
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
                     in_new_page=True, table_dfs=None, subfigs_dfs=None) -> None:

        """
        Agrega secciones a un PDF desde un DataFrame.

        Modos de interpretación del DataFrame:
        Título: La primera columna del DataFrame debe contener el título de la sección, comenzando con "#".
                Por ejemplo, "# Metodología Empleada".
        Texto normal: Sólo primera columna con texto, sin formato especial. Si el texto comienza con "*", se interpreta
                como texto en negrita.
        Texto en dos columnas: Si la segunda columna tiene el valor "two_columns", el texto de la primera columna se
                acumula hasta que se encuentra una fila que no tiene "two_columns" en la segunda columna.
        Texto condicional sin repetición: Si la segunda columna tiene "**", se interpreta como texto condicional a la
                condición dada por la tercera columna.
        Texto repetido y completado: Si la 2da columna tiene "$$", se interpreta como texto condicional a la condición
                dada por la tercera columna y se repite tantas veces como elementos cumplan la condición. Los valores
                solicitados en el texto encerrados por $$ se completan de los elementos disponibles.
        Imágenes: Si la primera columna comienza con "_fig" y la segunda columna contiene el nombre de un archivo de
                imagen, se interpreta como una figura a insertar. El caption de la figura se toma del texto que sigue
                a "_fig" en la primera columna, y el ancho de la imagen se toma de la tercera columna.
        Tablas: Si la primera columna comienza con "_tab", se inserta una tabla usando un DataFrame de table_dfs.


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
        table_dfs : list of DataFrames, optional
        """
        if in_new_page:
            self.add_page()
        if isinstance(df, list):
            df = pd.DataFrame(df)
        assert str(df.iloc[0, 0])[0] == "#", "La primera fila debe contener el título de la sección, comenzando con '#'"

        row = 0
        while row < len(df):
            if str(df.iloc[row, 0])[0:2] == "# ":
                predicted_text_y = (self.get_y() + vspace_before_title + self._pt2mm(font_size_title)
                    + vspace_from_title + self._pt2mm(font_size))
                if predicted_text_y > self.h - self.b_margin:
                    self.add_page()
                self.set_font(self.font, "B", size=font_size_title)
                self.ln(vspace_before_title)
                title = str(df.iloc[row, 0])[2:]
                alignment = "C" if center_title else "L"
                self.cell(0, 5, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=alignment)
                self.ln(vspace_from_title)
            elif str(df.iloc[row, 0])[0:4] == "_fig" and pd.notna(df.iloc[row, 1]):
                self.process_figure_request(df.iloc[row], subfigs_dfs=subfigs_dfs)
            elif str(df.iloc[row, 0])[0:4] == "_tab":
                ref, caption = self.get_ref_caption(str(df.iloc[row, 0]))
                assert table_dfs is not None, "Se encontró una fila de tabla pero no se proporcionó 'table_dfs'"
                if isinstance(table_dfs, list):
                    assert len(table_dfs) > self.counter_table, "No hay suficientes DataFrames para las tablas"
                    df_table = table_dfs[self.counter_table]
                else:
                    df_table = table_dfs
                self.counter_table += 1
                self.tables.append({"num": self.counter_table, "ref": ref})
                self.add_table_with_caption(df_table, caption, contains_multirows=True, extra_width=4, row_height=10)
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
                    for text in self.process_condition(str(df.iloc[row, 0]), str(df.iloc[row, 2])):
                        self.multi_cell(0, 5, text, border=0, align="J")
                        break
                elif df.shape[1] > 1 and df.iloc[row, 1] == "$$" and self.elements is not None:
                    for text in self.process_condition(str(df.iloc[row, 0]), str(df.iloc[row, 2])):
                        self.multi_cell(0, 5, text, border=0, align="J")
                elif not pd.isna(df.iloc[row, 1]):
                    self.write_value_with_units(df.iloc[row, 0], df.iloc[row, 1])
                else:
                    raise ValueError(f"Fila {row} tiene formato inesperado: '{df.iloc[row, 0]}', '{df.iloc[row, 1]}'")
                if vspace_after_text > 0 and writing:
                    self.ln(vspace_after_text)
            row += 1

    def process_condition(self, base_text: str, condition: str, behavior: str = "**"):
        key, value = condition.split(":", 1)
        key = key.strip()
        value = value.strip()
        self.process_elements(key)
        self.current_iteration_total = len(
            [x for x in self.elements_info[key] if x == value or (value == 'Any' and isinstance(x, str))]
        )
        n_this_value = self.current_iteration_total if behavior == "**" else None
        n_other_values = len([x for x in self.elements_info[key] if x != value and x is not None and value != "Any"])
        try:
            for k_index, element in enumerate(self.elements):
                self.current_iteration_element = k_index
                if key in element and (value in element[key] or value == "Any"):
                    yield self.insert_values_into_text(base_text, [element, self.general_info], n=n_this_value,
                                                       n_not=n_other_values)
        finally:
            self.current_iteration_total = None
            self.current_iteration_element = None

    def process_figure_request(self, df_row: pd.Series, subfigs_dfs=None):
        ref, caption = self.get_ref_caption(str(df_row.iloc[0]))
        path_or_subfigs = str(df_row.iloc[1])
        maybe_tuple = self._parse_tuple(path_or_subfigs)
        if maybe_tuple is None:
            file_path = self.full_resource_path(path_or_subfigs)
            subfigs_case = False
        else:
            if subfigs_dfs:
                assert len(subfigs_dfs) == len(self.elements), \
                    "Se encontró una figura con subfiguras pero no se proporcionó un 'subfigs_dfs' adecuado."
            subfigs_rows_cols = [int(x) for x in maybe_tuple]
            subfigs_case = True
        width_mm = df_row.iloc[2] if len(df_row) > 2 and pd.notna(df_row.iloc[2]) else None
        if len(df_row) > 3 and pd.notna(df_row.iloc[3]):
            for k_element, text in enumerate(self.process_condition(caption, str(df_row.iloc[3]))):
                self.counter_figure += 1
                if self.current_iteration_total == 1:
                    self.figures.append({"num": self.counter_figure, "ref": ref})
                else:
                    self.figures.append({"num": self.counter_figure, "ref": f"{ref}_{k_element + 1}"})
                if subfigs_case:
                    self.add_figure(subfigs_dfs[self.current_iteration_element], caption=text, width_mm=width_mm,
                                    subfigs_rows_cols=subfigs_rows_cols)
                else:
                    self.add_figure(file_path, caption=text, width_mm=width_mm)

    def full_resource_path(self, file_path):
        if os.path.isabs(file_path):
            return file_path
        else:
            return os.path.join(self.resource_path, file_path)

    @staticmethod
    def get_ref_caption(string: str):
        sep_index = string.find(" ")
        ref = string[:sep_index]
        caption = string[sep_index + 1:].strip()
        return ref, caption

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
        self.set_font(self.font, style="I", size=10)
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
                this_n_ln, _ = extras.estimate_lines(self, col_name, col_widths[col_k])
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

    def add_figure(self, filename, caption=None, width_mm=None, vspace_before_caption=2, vspace_after_figure=3,
                   subfigs_rows_cols=None, subfigs_hspace_ratio=0.05, subfigs_vspace_mm=2, subfigs_same_height=True,
                   caption_height=5):
        if width_mm is None or width_mm == 0:
            width_mm = self.effective_page_width  # Por defecto, ocupar el ancho efectivo
        x_center = self.l_margin + (self.effective_page_width - width_mm) / 2
        if isinstance(filename, str) and os.path.isfile(filename):
            info = get_img_info(filename)
            height_mm = width_mm * info.height / info.width + 2 * caption_height
            if self.get_y() + height_mm > self.h - self.b_margin:
                self.add_page()
            self.image(filename, x=x_center, w=width_mm)
        elif isinstance(filename, DataFrame) and subfigs_rows_cols is not None:
            subfigs_dfs = filename
            n_ims = len(subfigs_dfs)
            assert subfigs_rows_cols is not None and subfigs_rows_cols[0] * subfigs_rows_cols[1] >= n_ims, \
                "subfigs_rows_cols debe ser un par (n_rows, n_cols) tal que n_rows * n_cols >= n_ims"
            assert 0 < subfigs_hspace_ratio < 1, "subfigs_hspace_ratio debe estar entre 0 y 1"
            n_rows, n_cols = subfigs_rows_cols
            w_ims_coarse = width_mm / (n_cols + (n_cols - 1) * subfigs_hspace_ratio)
            hspace_between_bb = subfigs_hspace_ratio * w_ims_coarse
            w_ims, h_ims = [], []
            for i in range(n_ims):
                info = get_img_info(subfigs_dfs.at[i, "fullpath"])
                h_ims.append(info.height * w_ims_coarse / info.width)
                w_ims.append(info.width)
            h_ims_row = []
            for row in range(n_rows):
                h_ims_row.append(min(h_ims[row * n_cols:min(n_ims, (row + 1) * n_cols)]))
            height_mm = sum(h_ims_row) + 2 * subfigs_vspace_mm + self.font_size + 2 * caption_height
            if self.get_y() + height_mm > self.h - self.b_margin:
                self.add_page()
            for row, hrow in enumerate(h_ims_row):
                y_row = self.get_y()
                for col in range(n_cols):
                    if row * n_cols + col >= n_ims:
                        break
                    this_x = x_center + col * (w_ims_coarse + hspace_between_bb)
                    self.image(subfigs_dfs.at[row * n_cols + col, "fullpath"],
                               x=this_x,
                               y=y_row, w=w_ims_coarse, h=hrow, keep_aspect_ratio=True)
                    self.set_xy(this_x, y_row + hrow + subfigs_vspace_mm)
                    self.multi_cell(w_ims_coarse, caption_height, subfigs_dfs.at[row * n_cols + col, "subcaption"],
                                    align="C")
                self.ln(subfigs_vspace_mm)

        else:
            raise ValueError(
                "filename debe ser un archivo o una carpeta. subfigs_dfs y subfigs_rows_cols son necesarios si filename"
                " es una carpeta."
            )

        caption = f"Figura {self.counter_figure}: {caption}" if caption else f"Figura {self.counter_figure}"
        self.set_y(self.get_y() + vspace_before_caption)
        self.set_font(self.font, style="I", size=9)
        self.multi_cell(w=self.effective_page_width, h=caption_height, txt=caption, align="C", new_x="LMARGIN",
                        new_y="NEXT")
        self.ln(vspace_after_figure)

    def add_balanced_text(self, width_col=None, gutter=6, font_size=10, **kwargs):
        font_size_start = self.font_size_pt
        self.set_font(size=font_size)
        if width_col is None:
            width_col = (self.effective_page_width - gutter) / 2
        texto_izq, texto_der = extras.balance_text(self, self.buffer_two_columns, width_col=width_col, **kwargs)
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

    def insert_values_into_text(self, base_text, list_of_dicts, sep="$", sep_n="*", sep_not="/", n=1, n_not=0):
        """
            Replace:
                $<key>$ -> <value>
                *n == 1:n > 1*
                /only if n_not >= 1/
        """
        if isinstance(list_of_dicts, dict):
            list_of_dicts = [list_of_dicts]
        parts, types = self._parse_seps(base_text, (sep, sep_n, sep_not))
        for i in range(len(parts)):
            if types[i][-1] == sep:
                for d in list_of_dicts:
                    if parts[i] in self.key_aliases:
                        actual_key = self.key_aliases[parts[i]]
                        value = self.deep_get(d, actual_key)
                    elif parts[i] in d and d[parts[i]] is not None:
                        value = d[parts[i]]
                    else:
                        value = None
                    if value is not None:
                        parts[i] = str(value)
                        break
            if types[i][0] == sep_n:
                options = parts[i].split(":")
                assert len(options) == 2
                parts[i] = options[0] if n == 1 else options[1]
            if types[i][0] == sep_not:
                if n_not == 0:
                    parts[i] = ""
        return "".join(parts)

    @staticmethod
    def _parse_seps(text: str, seps: tuple):
        parts = []
        types = []
        this_str = ""
        this_type = "text"
        escape = False

        for char in text:
            if not escape and char == "\\":
                escape = True
                continue
            if char in seps and not escape:
                if this_str:
                    parts.append(this_str)
                    types.append(this_type)
                this_str = ""
                if this_type == "text":
                    this_type = char
                elif this_type[-1] == char:
                    this_type = this_type[:-1] if len(this_type) > 1 else "text"
                else:
                    this_type += char
            else:
                this_str += char
                if escape:
                    escape = False
        if this_type == "text" and this_str:
            parts.append(this_str)
            types.append(this_type)
        return parts, types

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

    @staticmethod
    def _parse_tuple(texto, n=2):
        texto = texto.strip()
        if not (texto.startswith("(") and texto.endswith(")")):
            return None
        contenido = texto[1:-1]
        partes = contenido.split(",")
        if n not in (None, 0) and len(partes) != n:
            return None
        try:
            t = tuple([p.strip() for p in partes])
            return t
        except ValueError:
            return None

    @staticmethod
    def _pt2mm(pt):
        return pt * 0.352875

    def append_element(self, element):
        if self.elements is None:
            self.elements = []
        self.elements.append(element)


if __name__ == "__main__":
    print(
        "Este módulo define la clase PDF para generar certificados.",
         " No se ejecuta nada al correr este archivo directamente."
    )
