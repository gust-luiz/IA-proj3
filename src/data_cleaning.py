from variables import MISS_DATA_TO_DROP_PERC

def clear_false_NAN_data(data_frame):
    return data_frame

def drop_more_blank_columns(data_frame):
    data_frame = clear_false_NAN_data(data_frame)

    to_drop = []
    cnt_row, _ = data_frame.shape

    for column in data_frame.columns:
        if data_frame[column].isna().value_counts().get(True, 0) / cnt_row >= MISS_DATA_TO_DROP_PERC:
            to_drop.append(column)

    return data_frame.drop(columns=to_drop)
