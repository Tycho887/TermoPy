# we want to read the "substances.JSON" file and create a dictionary with the properties of the substances

import json
import re

new_substances = {}

# we want to calculate the heat capacity of the substances at constant pressure and constant volume
# we want to calculate the enthalpy and the entropy of the substances

def heat_capacity(molar_mass,num_atoms):
    R = 8.31446261815324
    degrees_of_freedom = 3 + 2*(num_atoms-1)
    Cv = (degrees_of_freedom)/2*R
    Cp = Cv + R
    return Cv,Cp

def count_atoms(formula):
    # Split the formula into elements and their counts
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    # Create a dictionary to store element counts
    atom_counts = {}
    for element, count in elements:
        if count == '':
            count = 1
        else:
            count = int(count)
        if element in atom_counts:
            atom_counts[element] += count
        else:
            atom_counts[element] = count
    # we want to return the sum of atoms
    return sum(atom_counts.values())
    


with open("substances.JSON") as file:
    substances = json.load(file)
    # we want to remove the _ from the names of the substances
    for substance in substances:
        new_substances[substance.replace("_"," ")] = substances[substance]
    substances = new_substances


print(substances)

with open("substances.JSON","w") as file:
    json.dump(substances,file,indent=4)

