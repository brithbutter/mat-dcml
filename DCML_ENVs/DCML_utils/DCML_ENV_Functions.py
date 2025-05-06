# Worker performs functions with indipendent process
def worker_process( worker,workload,Pr,arrive_time):
    n_retry, delay, price = worker.process(workload,Pr,arrive_time)
    return n_retry, delay, price 

def calculate_reward(delay,payment):
    # alpha = 50
    # beta = 1
    # smooth_delay = (5**(delay+0.8))
    # if smooth_delay > 200:
        # smooth_delay = 200
    # return - smooth_delay*alpha - payment*beta
    # return - delay * alpha - math.sqrt(payment)*beta
    # return - ((delay+0.1)**1.2) * alpha - (payment**0.8)*beta
    alpha = 99
    beta = 1
    return - (delay*alpha) - (payment*beta)