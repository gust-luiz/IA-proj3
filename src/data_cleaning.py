from variables import MISS_DATA_TO_DROP_PERC


def clear_dataset(data_frame):
    '''Full cleaning process over a data frame

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `DataFrame:clean_df`
    '''
    data_frame = clear_false_NAN_data(data_frame)

    data_frame = _rename_columns(data_frame)
    data_frame = _drop_metacolumns(data_frame)
    data_frame = _drop_more_blank_columns(data_frame)

    return data_frame


def clear_false_NAN_data(data_frame):
    return data_frame


def _rename_columns(data_frame):
    '''Renaming dataset colums to facilitate data handle

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `Dataframe:updated_data_frame`
    '''
    return data_frame.rename(columns={
        'Patient ID': 'patient_id',
        'Patient age quantile': 'age_group',
        'SARS-Cov-2 exam result': 'has_covid19',
        # 'Patient addmited to regular ward (1=yes, 0=no)': '',
        # 'Patient addmited to semi-intensive unit (1=yes, 0=no)': '',
        # 'Patient addmited to intensive care unit (1=yes, 0=no)': '',
        # 'Hematocrit': '',
        # 'Hemoglobin': '',
        # 'Platelets': '',
        # 'Mean platelet volume ': '',
        # 'Red blood Cells': '',
        # 'Lymphocytes': '',
        # 'Mean corpuscular hemoglobin concentration (MCHC)': '',
        # 'Leukocytes': '',
        # 'Basophils': '',
        # 'Mean corpuscular hemoglobin (MCH)': '',
        # 'Eosinophils': '',
        # 'Mean corpuscular volume (MCV)': '',
        # 'Monocytes': '',
        # 'Red blood cell distribution width (RDW)': '',
        # 'Serum Glucose': '',
        # 'Respiratory Syncytial Virus': '',
        # 'Influenza A': '',
        # 'Influenza B': '',
        # 'Parainfluenza 1': '',
        # 'CoronavirusNL63': '',
        # 'Rhinovirus/Enterovirus': '',
        # 'Mycoplasma pneumoniae': '',
        # 'Coronavirus HKU1': '',
        # 'Parainfluenza 3': '',
        # 'Chlamydophila pneumoniae': '',
        # 'Adenovirus': '',
        # 'Parainfluenza 4': '',
        # 'Coronavirus229E': '',
        # 'CoronavirusOC43': '',
        # 'Inf A H1N1 2009': '',
        # 'Bordetella pertussis': '',
        # 'Metapneumovirus': '',
        # 'Parainfluenza 2': '',
        # 'Neutrophils': '',
        # 'Urea': '',
        # 'Proteina C reativa mg/dL': '',
        # 'Creatinine': '',
        # 'Potassium': '',
        # 'Sodium': '',
        # 'Influenza B, rapid test': '',
        # 'Influenza A, rapid test': '',
        # 'Alanine transaminase': '',
        # 'Aspartate transaminase': '',
        # 'Gamma-glutamyltransferase ': '',
        # 'Total Bilirubin': '',
        # 'Direct Bilirubin': '',
        # 'Indirect Bilirubin': '',
        # 'Alkaline phosphatase': '',
        # 'Ionized calcium ': '',
        # 'Strepto A': '',
        # 'Magnesium': '',
        # 'pCO2 (venous blood gas analysis)': '',
        # 'Hb saturation (venous blood gas analysis)': '',
        # 'Base excess (venous blood gas analysis)': '',
        # 'pO2 (venous blood gas analysis)': '',
        # 'Fio2 (venous blood gas analysis)': '',
        # 'Total CO2 (venous blood gas analysis)': '',
        # 'pH (venous blood gas analysis)': '',
        # 'HCO3 (venous blood gas analysis)': '',
        # 'Rods #': '',
        # 'Segmented': '',
        # 'Promyelocytes': '',
        # 'Metamyelocytes': '',
        # 'Myelocytes': '',
        # 'Myeloblasts': '',
        # 'Urine - Esterase': '',
        # 'Urine - Aspect': '',
        # 'Urine - pH': '',
        # 'Urine - Hemoglobin': '',
        # 'Urine - Bile pigments': '',
        # 'Urine - Ketone Bodies': '',
        # 'Urine - Nitrite': '',
        # 'Urine - Density': '',
        # 'Urine - Urobilinogen': '',
        # 'Urine - Protein': '',
        # 'Urine - Sugar': '',
        # 'Urine - Leukocytes': '',
        # 'Urine - Crystals': '',
        # 'Urine - Red blood cells': '',
        # 'Urine - Hyaline cylinders': '',
        # 'Urine - Granular cylinders': '',
        # 'Urine - Yeasts': '',
        # 'Urine - Color': '',
        # 'Partial thromboplastin time (PTT) ': '',
        # 'Relationship (Patient/Normal)': '',
        # 'International normalized ratio (INR)': '',
        # 'Lactic Dehydrogenase': '',
        # 'Prothrombin time (PT), Activity': '',
        # 'Vitamin B12': '',
        # 'Creatine phosphokinase (CPK) ': '',
        # 'Ferritin': '',
        # 'Arterial Lactic Acid': '',
        # 'Lipase dosage': '',
        # 'D-Dimer': '',
        # 'Albumin': '',
        # 'Hb saturation (arterial blood gases)': '',
        # 'pCO2 (arterial blood gas analysis)': '',
        # 'Base excess (arterial blood gas analysis)': '',
        # 'pH (arterial blood gas analysis)': '',
        # 'Total CO2 (arterial blood gas analysis)': '',
        # 'HCO3 (arterial blood gas analysis)': '',
        # 'pO2 (arterial blood gas analysis)': '',
        # 'Arteiral Fio2': '',
        # 'Phosphor': '',
        # 'ctO2 (arterial blood gas analysis)': '',
    })


def _drop_metacolumns(data_frame):
    '''Dropping columns with control data only aka metadata

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `Dataframe:updated_data_frame`
    '''
    return data_frame.drop(columns=[
        'patient_id',
    ])


def _drop_more_blank_columns(data_frame):
    '''Dropping columns with more blank data than filled data

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `Dataframe:updated_data_frame`
    '''
    to_drop = []
    cnt_row, _ = data_frame.shape

    for column in data_frame.columns:
        if data_frame[column].isna().value_counts().get(True, 0) / cnt_row >= MISS_DATA_TO_DROP_PERC:
            to_drop.append(column)

    return data_frame.drop(columns=to_drop)
