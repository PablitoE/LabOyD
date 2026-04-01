import os

import pandas as pd
from uncertainties import ufloat


class TemplateSheetReader:
    def __init__(self, df: pd.DataFrame, path_template=None):
        self.df = df
        self.path_template = path_template
        self.dir_template = os.path.dirname(path_template) if path_template is not None else None

    def get_calibration_date(self, value):
        if isinstance(value, str):
            string = value.replace("/", "-")
            return pd.to_datetime(string, dayfirst=True)
        elif isinstance(value, int):
            return pd.to_datetime(value, origin="1899-12-30", unit="D", dayfirst=True)


class SpecsReader(TemplateSheetReader):
    def __init__(self, df: pd.DataFrame, path_template=None):
        super().__init__(df, path_template)
        self.first_rows = ("TIPO TRABAJO", "Número", "Sub Trabajo", "Usuario", "Personal interviniente",
                           "LONGITUD DE ONDA", "Incertidumbre LAMBDA")
        self.assert_job_data()
        self.elements = []
        self.get_elements()

        self.diameter_bigger_than_reference = False
        self.check_diameters()
        self.add_general_data()

    def assert_job_data(self):
        for i, expected_value in enumerate(self.first_rows):
            assert self.df.iloc[i, 0].strip().startswith(expected_value), \
                f"La fila {i+1} de la hoja 'Especificaciones' debe contener '{expected_value}' en la primera columna."
        assert self.df.iloc[len(self.first_rows), 0].strip().startswith('TIPO MUESTRA'), (
            'Luego de los datos deben haber elementos ingresados según plantilla.'
        )

    def get_elements(self):
        new_element = None
        for i in range(len(self.first_rows), len(self.df)):
            if self.df.iloc[i, 0].strip() == 'TIPO MUESTRA':
                if new_element is not None:
                    self.elements.append(new_element)
                if self.df.iloc[i, 2] == 'No':
                    new_element = None
                elif self.df.iloc[i, 2] == 'Si':
                    if self.df.iloc[i, 1] in ('PLANO', 'JUEGO DE PARALELAS', 'PARALELA'):
                        new_element = {'tipo_muestra': self.df.iloc[i, 1]}
                    else:
                        raise ValueError(f"Valor inesperado en 'TIPO MUESTRA': {self.df.iloc[i, 1]}")
                else:
                    raise ValueError(f"Valor inesperado en USO de 'TIPO MUESTRA': {self.df.iloc[i, 2]}")
            elif new_element is not None:
                key = self.df.iloc[i, 0].strip().replace(' ', '_').replace(':', '')
                key = key.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                if key in new_element:
                    raise ValueError(f"Clave duplicada '{key}' para el mismo tipo de muestra.")
                if self.df.iloc[i, 0].strip().startswith('Identificación usuario'):
                    new_element[key] = self.df.iloc[i, 1:].dropna().tolist()
                    for index, value in enumerate(new_element[key]):
                        if isinstance(value, float) and value.is_integer():
                            new_element[key][index] = int(value)
                    new_element[key] = [str(x).strip() for x in new_element[key]]
                    if len(new_element[key]) == 1:
                        new_element[key] = new_element[key][0]
                else:
                    new_element[key] = self.df.iloc[i, 1]
        if new_element is not None:
            self.elements.append(new_element)
        return self.elements

    def check_diameters(self):
        for element in self.elements:
            if "Diametro" in element and "Diametro_referencia" in element:
                self.diameter_bigger_than_reference = element["Diametro"] > element["Diametro_referencia"]
                if self.diameter_bigger_than_reference:
                    break

    def add_general_data(self):
        self.general_info = {
            "diametro_menor": "Si" if self.diameter_bigger_than_reference else "No",
            "Longitud_de_onda": '{:.1up}'.format(self.get_lambda()).replace('+/-', ' ± ').replace('.', ','),
        }

    def has_element(self, key, value):
        for element in self.elements:
            if key in element and element[key] == value:
                return True
        return False

    def get_lambda(self):
        for _, row in self.df.iterrows():
            if row[0] == 'LONGITUD DE ONDA':
                value = row[1]
            if row[0] == 'Incertidumbre LAMBDA':
                uncertainty = row[1]
        return ufloat(value, uncertainty)

class FEIReader(TemplateSheetReader):
    def __init__(self, df: pd.DataFrame, path_template=None):
        super().__init__(df, path_template)
        self.calibration_date = self.get_calibration_date(self.df.iloc[1, 1])
        self.elements = self.get_elements()

    def get_elements(self):
        elements = []
        row = 2
        while row < self.df.shape[0]:
            if self.df.iloc[row, 0] == 'Identificación':
                element = {
                    "id": str(self.df.iloc[row + 1, 0]),
                    "cara_superior": self._read_face(row + 1),
                    "cara_inferior": self._read_face(row + 5)
                    }
                elements.append(element)
                row += 9
            elif self.df.iloc[row, 0].startswith('Tprom'):
                self.temperature = self.df.iloc[row, 1]
                self.humidity = self.df.iloc[row + 1, 1]
                row += 2
            else:
                row += 1
        return elements

    def _read_face(self, start_row):
        face_data = {
            "file_curve": self.df.iloc[start_row + 1, 4],
            "file_log": self.df.iloc[start_row + 2, 4],
            "n_ims": self.df.iloc[start_row + 3, 4],
            "temperature": self.df.iloc[start_row, 6],
            "humidity": self.df.iloc[start_row + 1, 6],
            "initial_observations": self.df.iloc[start_row, 10],
            "final_observations": self.df.iloc[start_row + 1, 10],
            "value": self.df.iloc[start_row + 1, 8],
            "uncertainty": self.df.iloc[start_row + 3, 8]
        }
        return face_data

    def merge_with_elements(self, elements, id_key):
        new_elements = elements.copy()
        for element in new_elements:
            assert isinstance(element, dict), "Cada elemento debe ser un diccionario"
            if id_key not in element:
                continue
            fei_data = []
            if isinstance(element[id_key], str):  # If the id_key is a string, it's a single element
                element[id_key] = [element[id_key]]
            for element_id in element[id_key]:
                for fei_element in self.elements:
                    if element_id == fei_element['id']:
                        fei_data.append({
                            "Cara_superior": fei_element['cara_superior'],
                            "Cara_inferior": fei_element['cara_inferior']
                        })
                        break
                else:
                    fei_data.append(None)
            element['planitud_fei'] = fei_data[0] if len(fei_data) == 1 else fei_data
            element[id_key] = element[id_key][0] if len(element[id_key]) == 1 else element[id_key]
        return new_elements

    @staticmethod
    def build_table(elements):
        df = pd.DataFrame(columns=["Marca/Modelo", "Identificación", "Cara", "Desviación de planitud / nm"])
        for element in elements:
            key_id = 'Identificacion_usuario'
            if key_id not in element:
                continue
            sub_elements_ids = element[key_id] if isinstance(element[key_id], list) else [element[key_id]]
            key_fei = 'planitud_fei'
            sub_elements_fei = element[key_fei] if isinstance(element[key_fei], list) else [element[key_fei]]
            for k_sub_element in range(len(sub_elements_ids)):
                if sub_elements_fei[k_sub_element] is not None:
                    cara_superior = ufloat(sub_elements_fei[k_sub_element]['Cara_superior']['value'],
                                           sub_elements_fei[k_sub_element]['Cara_superior']['uncertainty'])
                    cara_inferior = ufloat(sub_elements_fei[k_sub_element]['Cara_inferior']['value'],
                                           sub_elements_fei[k_sub_element]['Cara_inferior']['uncertainty'])
                    new_rows = [
                        {
                            'Marca/Modelo': f'_{element["Marca"]} {element["Code"]}',
                            'Identificación': f'_{sub_elements_ids[k_sub_element]}',
                            'Cara': 'Superior',
                            'Desviación de planitud / nm': '{:.2up}'.format(cara_superior)
                            .replace('+/-', ' ± ').replace('.', ','),
                        },
                        {
                            'Marca/Modelo': f'_{element["Marca"]} {element["Code"]}',
                            'Identificación': f'_{sub_elements_ids[k_sub_element]}',
                            'Cara': 'Inferior',
                            'Desviación de planitud / nm': '{:.2up}'.format(cara_inferior)
                            .replace('+/-', ' ± ').replace('.', ','),
                        },
                    ]
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df


class ToleranceReader(TemplateSheetReader):
    def __init__(self, df: pd.DataFrame, path_template=None):
        super().__init__(df, path_template)
        self.calibration_date = self.get_calibration_date(self.df.iloc[1, 1])
        self.dir = []
        self.elements = self.get_elements()

    def get_elements(self):
        elements = []
        row = 2
        box = 0
        while row < self.df.shape[0]:
            if str(self.df.iloc[row, 3]).strip() == 'Cara superior':
                element = {
                    "box": box,
                    "id": str(self.df.iloc[row, 0]),
                    "Cara_superior": self._read_face(row),
                    "Cara_inferior": self._read_face(row + 4),
                    }
                elements.append(element)
                row += 8
            elif str(self.df.iloc[row, 0]).startswith('Tprom'):
                self.temperature = self.df.iloc[row, 1]
                self.humidity = self.df.iloc[row + 1, 1]
                row += 2
            elif str(self.df.iloc[row, 0]).startswith('Directorio'):
                self.dir.append(self.df.iloc[row, 1])
                if self.dir[-1][0] != os.sep:
                    self.dir[-1] = os.path.join(self.dir_template, self.dir[-1])
                assert os.path.isdir(self.dir[-1]), f"El directorio {self.dir[-1]} no existe"
                box += 1
                row += 1
            else:
                row += 1
        return elements

    def _read_face(self, row):
        return {
            "archivo": str(self.df.iloc[row + 1, 5]),
            "temperatura": float(self.df.iloc[row, 7]),
            "humedad": float(self.df.iloc[row + 2, 7])
        }

    def prepare_subfigs(self, all_elements):
        elements_ids = [element['id'] for element in self.elements]
        dfs = []
        for element in all_elements:
            if "Metodo_planitud" not in element or element["Metodo_planitud"] != "Tolerancia":
                dfs.append(None)
                continue
            df = pd.DataFrame(columns=["Identificación", "subcaption", "fullpath"])
            key_id = 'Identificacion_usuario'
            sub_elements_ids = element[key_id] if isinstance(element[key_id], list) else [element[key_id]]
            for k_sub_element in range(len(sub_elements_ids)):
                id = sub_elements_ids[k_sub_element]
                element_index = elements_ids.index(id)
                dirpath = self.dir[self.elements[element_index]['box']]
                files_in_dir = os.listdir(dirpath)
                basenames = [os.path.splitext(file)[0] for file in files_in_dir]
                if self.elements[element_index]['Cara_superior']['archivo'].endswith('.png'):
                    list_for_search = files_in_dir
                else:
                    list_for_search = basenames
                file_sup = files_in_dir[list_for_search.index(self.elements[element_index]['Cara_superior']['archivo'])]
                file_inf = files_in_dir[list_for_search.index(self.elements[element_index]['Cara_inferior']['archivo'])]
                new_rows = [{
                    "Identificación": f"{id}",
                    "cara": "Superior",
                    "subcaption": f"{id} (a)",
                    "fullpath": os.path.join(dirpath, file_sup)
                }, {
                    "Identificación": f"{id}",
                    "cara": "Inferior",
                    "subcaption": f"{id} (b)",
                    "fullpath": os.path.join(dirpath, file_inf)
                }]
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            dfs.append(df)
        return dfs


class ParallelismReader(TemplateSheetReader):
    def __init__(self, df_values: pd.DataFrame, df_uncertainties: pd.DataFrame, path_template=None):
        super().__init__(df_values, path_template)
        self.df_values = df_values
        self.df_uncertainties = df_uncertainties
        self.calibration_date = self.get_calibration_date(self.df_values.iloc[2, 1])
        self.use_set = self.df_values.iloc[2, 16]
        self.boxes, self.elements = self.get_elements()

    def get_index_by_id(self, id):
        for index, element in enumerate(self.elements):
            if element['id'] == id:
                return index
        return None

    def get_ufloat(self, index: int):
        set = 'setA' if self.use_set == 'A' else 'setB'
        return ufloat(self.elements[index][set]['deviation'], self.elements[index]['uncertainty'])

    def get_elements(self):
        elements = []
        row = 6
        box = []
        box_elements = []
        while row < self.df_values.shape[0]:
            if 'Identificación' in str(self.df_values.iloc[row, 0]):
                element = {
                    "id": str(self.df_values.iloc[row + 1, 0]),
                    "setA": self._read_measurement(row + 1, 1),
                    "setB": self._read_measurement(row + 1, 13),
                    "box": len(box)
                    }
                box_elements.append(element["id"])
                elements.append(element)
                row += 9
            elif str(self.df_values.iloc[row, 0]).startswith('Tprom'):
                box.append({
                    "elements": box_elements.copy(),
                    "temperature": self.df_values.iloc[row, 1],
                    "humidity": self.df_values.iloc[row + 1, 1]
                })
                box_elements = []
                row += 2
            else:
                row += 1

        row = 0
        element = 0
        while row < self.df_uncertainties.shape[0]:
            if 'repetibilidad' in str(self.df_uncertainties.iloc[row, 0]):
                elements[element]["uncertainty"] = float(self.df_uncertainties.iloc[row + 6, 9]) / 1e3  # nm to µm
                element += 1
                row += 6
            else:
                row += 1
        if element != len(elements):
            raise ValueError("No se encontraron las incertidumbres de todos los elementos.")
        return box, elements

    def _read_measurement(self, row, col):
        dev = float(self.df_values.iloc[row + 7, col + 2])
        return {
            "deviation": dev,
        }

    def build_table(self, elements):
        df = pd.DataFrame(columns=["Marca/Modelo", "Identificación", "Desviación de paralelismo / µm"])
        for element in elements:
            if "Paralelismo" not in element or element["Paralelismo"] != "Si":
                continue
            if element["Paralelismo"] == "Si":
                key_id = 'Identificacion_usuario'
                sub_elements_ids = element[key_id] if isinstance(element[key_id], list) else [element[key_id]]
                n_subelements = len(sub_elements_ids)
                new_rows = []
                for k_sub_element in range(len(sub_elements_ids)):
                    if self.elements[k_sub_element] is not None:
                        paralelismo = self.get_ufloat(self.get_index_by_id(sub_elements_ids[k_sub_element]))
                        marca = (
                            f'{"_" * (n_subelements - 1)}{element["Marca"]} {element["Code"]}'
                            if k_sub_element == 0 else ''
                        )
                        new_rows.append({
                            "Marca/Modelo": marca,
                            "Identificación": f"{sub_elements_ids[k_sub_element]}",
                            "Desviación de paralelismo / µm": "{:.2fp}".format(paralelismo)
                            .replace("+/-", " ± ").replace(".", ",")
                        })
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df
