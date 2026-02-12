import logging
import os

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pandas import DataFrame

from src.Certificados.pdf_extras import balance_text

logger = logging.getLogger(__name__)


class PDF(FPDF):
    def __init__(self, datos_ot_rut, resource_path="resources/Certificados", font="helvetica"):
        super().__init__()
        self.datos_ot_rut = datos_ot_rut
        self.resource_path = resource_path
        self.font = font
        self.alias_nb_pages()  # Funcion que hay que llamar para poder numerar las paginas tipon "1 de"
        self.set_auto_page_break(auto=True, margin=30)
        self.set_top_margin(50)

        self.buffer_two_columns = ""  # Para almacenar temporalmente el texto de dos columnas

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

    def add_sections(self, df: DataFrame, vspace_from_title=2, vspace_after_text=0, center_title=False, font_size=10,
                     font_size_title=12, font_size_two_columns=8.5, in_new_page=True) -> None:
        if in_new_page:
            self.add_page()
        if isinstance(df, list):
            df = pd.DataFrame(df)
        assert str(df.iloc[0, 0])[0] == "#", "La primera fila debe contener el título de la sección, comenzando con '#'"
        row = 0
        while row < len(df):
            if str(df.iloc[row, 0])[0:2] == "# ":
                self.set_font(self.font, "B", size=font_size_title)
                title = str(df.iloc[row, 0])[2:]
                alignment = "C" if center_title else "L"
                self.cell(0, 5, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=alignment)
                self.ln(vspace_from_title)
            elif str(df.iloc[row, 0])[0:4] == "_fig" and pd.notna(df.iloc[row, 1]):
                caption = str(df.iloc[row, 0])[4:].strip()
                file_path = os.path.join(self.resource_path, str(df.iloc[row, 1]))
                width_mm = df.iloc[row, 2] if df.shape[1] > 2 and pd.notna(df.iloc[row, 2]) else None
                self.add_figure(file_path, caption=caption, width_mm=width_mm)
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
                else:
                    self.write_value_with_units(df.iloc[row, 0], df.iloc[row, 1])
                if vspace_after_text > 0 and writing:
                    self.ln(vspace_after_text)
            row += 1

    def write_value_with_units(self, value, html_units):
        expansion = f"{value} {html_units}"
        self.write_html(expansion)
        self.ln(10)  # write_html no da salto de línea automático al final, hay que agregarlo

    def add_table_with_caption(
        self, df: DataFrame, caption: str, caption_height=10, vspace_after_table=3, fit_col_widths=True,
        padding_lr=2, padding_tb=0, extra_width=0, row_height=5
    ):
        self.set_font(self.font, "", size=10)
        self.cell(self.effective_page_width, caption_height, caption, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                  align="C")
        if fit_col_widths:
            col_widths = []
            for col in df.columns:
                max_content_width = self.get_string_width(str(col))
                for val in df[col]:
                    val_width = self.get_string_width(str(val))
                    if val_width > max_content_width:
                        max_content_width = val_width
                col_widths.append(max_content_width + 2 * padding_lr)  # Agregar un poco de padding
            total_width = sum(col_widths) + extra_width * len(df.columns)  # Agregar espacio para los bordes
            total_width = self.effective_page_width if total_width > self.effective_page_width else total_width
        else:
            col_widths = None
            total_width = self.effective_page_width

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
                for valor in fila:
                    row.cell(str(valor))
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
