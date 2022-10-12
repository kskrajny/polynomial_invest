from datetime import timedelta

from get_data import show_sample_data

delta_list = [
    timedelta(hours=1),
    timedelta(minutes=20),
    timedelta(minutes=5)
]

show_sample_data(delta_list=delta_list)
