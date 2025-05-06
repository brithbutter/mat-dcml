class Action_Space():
    def __init__(self,dim,semi_index = 0,extra=False,multi_discrete=False,high=None, low = None,mixed = True,continuous=True) -> None:
        self.continuous = continuous
        self.mixed = mixed
        self.n = dim
        self.extra = extra
        self.semi_index = semi_index
        self.multi_discrete = multi_discrete
        self.high = high
        self.low = low