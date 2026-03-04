import numpy as np


def extract_features(keystrokes):
    dwell_times = []
    flight_times = []

    # Consider only keystrokes that have both press and release timestamps
    valid_keystrokes = [
        k for k in keystrokes if "down" in k and "up" in k
    ]

    if len(valid_keystrokes) < 2:
        # Poucos dados para extrair um padrão significativo
        return np.array([])

    for i in range(len(valid_keystrokes)):
        dwell = valid_keystrokes[i]["up"] - valid_keystrokes[i]["down"]
        dwell_times.append(dwell)

        if i > 0:
            flight = (
                valid_keystrokes[i]["down"] - valid_keystrokes[i - 1]["up"]
            )
            flight_times.append(flight)

    dwell = np.array(dwell_times)
    flight = np.array(flight_times) if flight_times else None

    # Estatísticas de dwell
    dwell_mean = float(dwell.mean())
    dwell_std = float(dwell.std()) if dwell.size > 1 else 0.0

    # Estatísticas de flight (se existirem)
    if flight is not None and flight.size > 0:
        flight_mean = float(flight.mean())
        flight_std = float(flight.std()) if flight.size > 1 else 0.0
    else:
        flight_mean = 0.0
        flight_std = 0.0

    # Vetor de características de tamanho fixo (4)
    return np.array([dwell_mean, dwell_std, flight_mean, flight_std])
