import numpy as np
import itertools
print(np.array(list(itertools.product([0, 1], repeat=2))))

print(np.array(['up','down','left','right']))

print(np.zeros(2,dtype='int'))

print(np.append([1,1],np.zeros(2,dtype='int')))

free_locations = []
free_locations.append([2,1])
free_locations.append([1,2])
free_locations = np.array(free_locations)
print(free_locations)

for free_location in free_locations:
    print(free_location)

print(np.concatenate([[2,1],[0,0]],0))

def value_of_capital_letter(letter):
    ''' Gives an index to a capital letter, i.e. 'A' -> 1, 'B' -> 2 '''
    return ord(letter) - ord('A') + 1
print(value_of_capital_letter('C'))

def value_of_lower_letter(letter):
    ''' Gives an index to a capital letter, i.e. 'a' -> 1, 'b' -> 2 '''
    return ord(letter) - ord('a') + 1