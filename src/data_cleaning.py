import numpy as np
from pandas import concat, get_dummies

from utils import to_snake_case
from variables import MISS_DATA_TO_DROP_PERC


def clear_dataset(data_frame):
    '''Full cleaning process over a data frame

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `DataFrame:clean_df`
    '''
    data_frame = _rename_columns(data_frame)
    data_frame = _replace_humanized_values(data_frame)
    data_frame = _categorical_to_number(data_frame)
    #data_frame = fill_NAN_fields_zero(data_frame)
    #data_frame = fill_NAN_fields_mean(data_frame)
    #data_frame = fill_NAN_fields_group_mean(data_frame)  #overfitting

    data_frame = _drop_metacolumns(data_frame)
    data_frame = _drop_more_blank_columns(data_frame)

    data_frame = _drop_more_blank_lines(data_frame)

    return data_frame


def _replace_humanized_values(data_frame):
    # Replacing humanized values with 0, 1 and nan
    data_frame = data_frame.replace(
        ['positive', 'detected', 'present'],
        1
    )
    data_frame = data_frame.replace(
        ['negative', 'not_detected', 'normal', 'absent', 'Ausentes'],
        0
    )
    data_frame = data_frame.replace(
        ['not_done',  'Não Realizado'],
        np.nan
    )

    # Replacing inequalities with number and defining columns datatype to float where it concerns
    data_frame['urine_leukocytes'] = data_frame['urine_leukocytes'].replace('<1000', '999', inplace=True)
    data_frame['urine_leukocytes'] = data_frame['urine_leukocytes'].astype("float64")
    data_frame['urine_ph'] = data_frame['urine_ph'].astype('float64')

    return data_frame


def _categorical_to_number(data_frame):
    # Get dataframe without dtype object, except column "Patient ID"
    data_not_object = concat(
        [data_frame['patient_id'], data_frame[data_frame.dtypes[data_frame.dtypes != "object"].index]],
        axis='columns'
    )

    # Get dataframe with dummies
    data_dummies = get_dummies(data_frame[data_frame.dtypes[data_frame.dtypes == "object"].drop('patient_id').index])

    # Concatenate dataframe with dummies
    data_frame = concat([data_not_object, data_dummies], axis='columns')

    return data_frame


def fill_NAN_fields_zero(data_frame):
    return data_frame.fillna(0)


def fill_NAN_fields_mean(data_frame):
    return data_frame.fillna(data_frame.mean())


def _rename_columns(data_frame):
    '''Renaming dataset colums to facilitate data handle

    Args:
    - `DataFrame:data_frame`: Data frame to be clear

    Returns:
    - `Dataframe:updated_data_frame`
    '''
    data_frame = data_frame.rename(columns={
        'Patient age quantile': 'Age Group',
        'SARS-Cov-2 exam result': 'Has COVID-19',
        'pCO2 (venous blood gas analysis)': 'Venous pCO2',
        'Hb saturation (venous blood gas analysis)': 'Venous Hb saturation',
        'Base excess (venous blood gas analysis)': 'Venous Base excess',
        'pO2 (venous blood gas analysis)': 'Venous pO2',
        'Fio2 (venous blood gas analysis)': 'Venous Fio2',
        'Total CO2 (venous blood gas analysis)': 'Venous Total CO2',
        'pH (venous blood gas analysis)': 'Venous pH',
        'HCO3 (venous blood gas analysis)': 'Venous HCO3',
        'pCO2 (arterial blood gas analysis)': 'Arterial pCO2',
        'Hb saturation (arterial blood gases)': 'Arterial Hb saturation',
        'Base excess (arterial blood gas analysis)': 'Arterial Base excess',
        'pH (arterial blood gas analysis)': 'Arterial pH',
        'Total CO2 (arterial blood gas analysis)': 'Arterial Total CO2',
        'HCO3 (arterial blood gas analysis)': 'Arterial HCO3',
        'pO2 (arterial blood gas analysis)': 'Arterial pO2',
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
        # 'Arteiral Fio2': '',
        # 'Phosphor': '',
        # 'ctO2 (arterial blood gas analysis)': '',
    })

    return data_frame.rename(to_snake_case, axis='columns')


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


def _drop_more_blank_lines(data_frame):
    min_positive_filled = data_frame.loc[data_frame['has_covid_19'] == 1].count(axis='columns').min()

    negative_filled = data_frame.loc[data_frame['has_covid_19'] == 0].count(axis='columns')
    negative_filled = negative_filled > round(data_frame.shape[1] * (1 - min_positive_filled / data_frame.shape[1]))

    data_frame = data_frame.drop(index=negative_filled.loc[negative_filled.values == False].index)

    # print(data_frame['has_covid_19'].value_counts())
    # input()

    return data_frame

# Function to visualize columns dtype object for posterior data cleaning
def analyze_object_columns(data_frame):
    object_columns = []

    # Get columns with dtype object
    for c in data_frame.columns:
        column_dtype = data_frame[c].dtype
        if column_dtype == 'object' and c != 'patient_id':
            object_columns.append(c)

    print(object_columns)
    print()

    # Show unique values in object columns to analyze
    for oc in object_columns:
        print('Object Column: ', oc)
        print(data_frame[oc].unique())
        print()
