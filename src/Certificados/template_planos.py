import pandas as pd


class FEIReader:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.calibration_date = pd.to_datetime(self.df.iloc[1, 1], origin="1899-12-30", unit="D", dayfirst=True)
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
            sub_elements_ids = element[key_id] if isinstance(element[key_id], list) else [element[key_id]]
            key_fei = 'planitud_fei'
            sub_elements_fei = element[key_fei] if isinstance(element[key_fei], list) else [element[key_fei]]
            for k_sub_element in range(len(sub_elements_ids)):
                if sub_elements_fei[k_sub_element] is not None:
                    new_rows = [{
                        "Marca/Modelo": f"_{element['Marca']} {element['Code']}",
                        "Identificación": f"_{sub_elements_ids[k_sub_element]}",
                        "Cara": "Superior",
                        "Desviación de planitud / nm": f"({sub_elements_fei[k_sub_element]['Cara_superior']['value']}"
                        f" ± {sub_elements_fei[k_sub_element]['Cara_superior']['uncertainty']})"
                    }, {
                        "Marca/Modelo": f"_{element['Marca']} {element['Code']}",
                        "Identificación": f"_{sub_elements_ids[k_sub_element]}",
                        "Cara": "Inferior",
                        "Desviación de planitud / nm": f"({sub_elements_fei[k_sub_element]['Cara_inferior']['value']}"
                        f" ± {sub_elements_fei[k_sub_element]['Cara_inferior']['uncertainty']})"
                    }]
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df
