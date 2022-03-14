from make_campaign import optimize
from make_simulator import make_dataset
# required arguments #

name = 'custom'
algorithm = 'Phoenics'
goal = 'maximize'
max_iter = 4
campaign_number = 1

print('required arguments:\n')
print('name: {} (dataset name)'.format(name))
print('algorithm: {} (optimization algorithm)'.format(algorithm))
print('goal: {} (objective function goal)'.format(goal))
print('campaign_number = {} (campaign number)'.format(campaign_number))
print('max_iter = {} (maximum iterations (number of observations generated))'.format(max_iter))

dataset = make_dataset(name)
optimize(dataset,name,algorithm,goal,max_iter,campaign_number)
