def compute_pcr(options):
    total_call_oi = 0
    total_put_oi = 0

    for opt in options:
        if opt["instrument_name"].endswith("C"):
            total_call_oi += opt["open_interest"]
        else:
            total_put_oi += opt["open_interest"]

    return round(total_put_oi / total_call_oi, 3)
