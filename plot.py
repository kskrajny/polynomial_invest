from main import delta_list, polynomial_degree, future_length, tau, instrument_list, candle_num, start_date,\
    final_date
from get_data import show_sample_data

print(instrument_list)
show_sample_data(instrument_name=instrument_list[0], date_from=start_date, date_to=final_date,
                 tau=tau, future_length=future_length, delta_list=delta_list, polynomial_degree=polynomial_degree,
                 candle_num=candle_num)
