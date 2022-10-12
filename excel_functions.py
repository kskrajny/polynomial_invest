def save_text(writer, text):
    worksheet = writer.sheets['Sheet1']

    options = {
        'height': 400,
        'width': 150
    }

    worksheet.insert_textbox('I2', text, options)
    writer.save()


def create_meta_text(instrument_list, delta_list, data_delta, lr):
    text = "Instruments:"
    for name in instrument_list:
        name = name.split('_')[0]
        text += "\n" + name
    text += "\n\nDeltas:"
    for delta in delta_list:
        delta = str(delta).split('.')[0]
        text += "\n" + delta
    text += "\n\nData delta:\n" + str(data_delta).split('.')[0]
    text += "\n\nlr:\n" + str(lr)
    return text
