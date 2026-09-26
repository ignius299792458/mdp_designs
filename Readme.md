# MDPs Designs : Markov Decision Processes System Designs

1. Recycling Robot: it is simple recycling robot MDP designs:
   - Goal : "to collect as many cans as possible without draining the battery"
   - states: {battery_high, battery_low}
   - action: {search, wait, recharge}
   - reward: {set of maps r(s,a,s')->certain_value}
