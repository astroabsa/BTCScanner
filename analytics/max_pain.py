def compute_max_pain(options):
    strikes = sorted(set([opt["strike"] for opt in options]))
    pain = {}

    for strike in strikes:
        total_pain = 0
        for opt in options:
            oi = opt["open_interest"]
            if opt["instrument_name"].endswith("C"):
                total_pain += max(0, strike - opt["strike"]) * oi
            else:
                total_pain += max(0, opt["strike"] - strike) * oi
        pain[strike] = total_pain

    return min(pain, key=pain.get)
