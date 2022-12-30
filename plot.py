from datetime import datetime
from contans_main import delta_list, polynomial_degree, future_length, tau, instrument_list, candle_num
from get_data import show_sample_data


show_sample_data(instrument_name=instrument_list[0], date_from=datetime(2014, 1, 1), date_to=datetime(2014, 12, 28),
                 tau=tau, future_length=future_length, delta_list=delta_list, polynomial_degree=polynomial_degree,
                 candle_num=candle_num)
