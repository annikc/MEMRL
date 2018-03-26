import time

def run_reset(env, mf, ec): 
	global runtime 
	global blocktime
	global time_since_last_reward
	global no_r
	global cs_max
	global reward 

	ec.reset_cache()
	no_r 					= True
	time_since_last_reward	= 0
	cs_max					= 0
	reward 					= 0 

	# record current time before beginning of trial
	print "Run started: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
	runtime = time.time()
	blocktime = time.time()

def run_nothing():
	global a
	global b 

	a = 0
	b = 'string'
	return 

def trial_reset():
	global env
	trial_start = time.time()
	init_state 	= env.reset()
	reward_sum 	= 0

	return trial_start, init_state, reward_sum