config = {
    "mapping": {
        20: [i for i in range(20)],
        22: [i for i in range(22)],
        24: [i for i in range(24)],
        26: [i for i in range(26)],
    },
    # meas, prep
    "cuts": {
        20: [((0, 10), (1,0)), ((1, 0), (0,11))],
        22: [((0, 12), (1,0)), ((1, 0), (0,13))],
        24: [((0, 12), (1,0)), ((1, 0), (0,13))]
    }
}