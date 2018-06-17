#EX1
# Example, do not modify!

print(5 / 8)

# Put code below here

print(7+10)
##############################################

#EX2

#All of above
##############################################

#EX3

# Just testing division
print(5 / 8)

#Addition works too
print(7 + 10)
##############################################

#EX4
print(5 + 5)
print(5 - 5)

# Multiplication and division
print(3 * 5)
print(10 / 2)

# Exponentiation
print(4 ** 2)

# Modulo
print(18 % 7)

# How much is your $100 worth after 7 years?
print(100*(1.1**7))
##############################################

#EX5
# Create a variable savings
savings = 100

# Print out savings
print(savings)

##############################################

#EX6
# Create a variable savings
savings = 100

# Create a variable factor

factor = 1.1
# Calculate result

result = savings * factor**7
# Print out result
print(result)

#############################################

#EX7

# Create a variable desc
desc = "compound interest"

# Create a variable profitable
profitable = True

#############################################

#EX8

#a is float, b is str, c is bool

#############################################

#EX9

# Several variables to experiment with
savings = 100
factor = 1.1
desc = "compound interest"

# Assign product of factor and savings to year1
year1 = savings * factor

# Print the type of year1

print(type(year1))
# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(doubledesc)

#############################################

#EX10

# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

############################################

#EX11

#"The correct answer to this multiple choice exercise is answer number " + 2

############################################

#EX12

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall,kit,liv,bed,bath]

# Print areas
print(areas)


###########################################

#EX13

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway",hall,"kitchen", kit, "living room", liv,"bedroom", bed, "bathroom", bath]

# Print areas
print(areas)

#########################################

#EX14

#A,B and C

#########################################

#EX15

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom",bed],
         ["bathroom",bath]]

# Print out house
print(house)
print(type(house))

########################################

#EX16
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas

print(areas[1])
# Print out last element from areas

print(areas[-1])
# Print out the area of the living room
print(areas[5])

######################################

#EX17

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area

eat_sleep_area = areas[3]+areas[-3]

# Print the variable eat_sleep_area
print(eat_sleep_area)

######################################

#EX18
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs

downstairs = areas[0:6]
# Use slicing to create upstairs
upstairs = areas[6:]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)


######################################

#EX19
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[6:]

##########################################

#EX20
#A float: the bathroom area

##########################################

#EX21

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"

########################################

#EX22

# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]

########################################

#EX23

#del(areas[-4:-2])

##########################################

#EX24

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = areas[:]

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)

##########################################

#EX25

# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True

# Print out type of var1
print(type(var1))

# Print out length of var1

print(len(var1))
# Convert var2 to an integer: out2
out2 = int(var2)

################################################

#EX26

#complex() takes two arguments: real and imag.
#real is a required argument, imag is an optional argument.

##################################################

#EX27

# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full

full = first + second
# Sort full in descending order: full_sorted

full_sorted = sorted(full,reverse = True)
# Print out full_sorted
print(full_sorted)

#EX28

# string to experiment with: room
room = "poolhouse"

# Use upper() on room: room_up
room_up = room.upper()

# Print out room and room_up

print(room)
print(room_up)
# Print out the number of o's in room
print(room.count("o"))

##################################################

#EX29

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0

print(areas.index(20.0))
# Print out how often 14.5 appears in areas

print(areas.count(14.5))

######################################################


#EX30

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size

areas.append(24.5); areas.append(15.45)

# Print out areas

print(areas)
# Reverse the orders of the elements in areas

areas.reverse()
# Print out areas

print(areas)

######################################################

#EX31

# Definition of radius
r = 0.43

# Import the math package

import math as m

# Calculate C
C = 2*m.pi*r

# Calculate A
A = m.pi*r**2

# Build printout
print("Circumference: " + str(C))


######################################################

#EX32

# Definition of radius
r = 192500
phi = 12
# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
dist = r*radians(phi)

# Print out dist
print(dist)

#EX33

#from scipy.linalg import inv as my_inv

##########################################################


#EX34

# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))


############################################################


#EX35

# height is available as a regular list

# Import numpy
import numpy as np

# Create a numpy array from height: np_height

np_height = np.array(height)
# Print out np_height
print(np_height)

# Convert np_height to m: np_height_m
np_height_m = np_height * 0.0254

# Print np_height_m
print(np_height_m)

############################################################

#EX36

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254

# Create array from weight with correct units: np_weight_kg

np_weight_kg = np.array(weight) * 0.453592
# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m**2

# Print out bmi
print(bmi)

##########################################################

#EX37

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi<21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])

############################################################

#EX38

#np.array([4, 3, 0]) + np.array([0, 2, 2])

##############################################################

#EX39

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Store weight and height lists as numpy arrays
np_weight = np.array(weight)
np_height = np.array(height)

# Print out the weight at index 50
print(np_weight[50])

# Print out sub-array of np_height: index 100 up to and including index 110
print(np_height[100:111])

################################################################

#EX40

# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball

np_baseball = np.array(baseball)
# Print out the type of np_baseball

print(type(np_baseball))
# Print out the shape of np_baseball
print(np.shape(np_baseball))

##################################################################

#EX41

# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball

print(np.shape(np_baseball))

##################################################################

#EX42

# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball

print(np_baseball[49])
# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])


####################################################################

#EX43
# baseball is available as a regular list of lists
# updated is available as 2D numpy array

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1.])


# Print out product of np_baseball and conversion
print(np_baseball*conversion)

################################################################

#EX44

# np_baseball is available

# Import numpy
import numpy as np

# Create np_height from np_baseball
np_height = np_baseball[:,0]

# Print out the mean of np_height
print(np.mean(np_height))

# Print out the median of np_height
print(np.median(np_height))

################################################################

#EX45

# np_baseball is available

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))

################################################################

#EX46

# heights and positions are available as lists

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_heights = np.array(heights)
np_positions = np.array(positions)

# Heights of the goalkeepers: gk_heights

gk_heights = np_heights[np_positions == 'GK']
# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))

##################################################################
#END
