# we want to read the data in the file "substances.txt" and create the dictionary "substances" automatically

# the information in the file is in the following format:
# Acetone	(CH3)2CO	1.47	1.32	0.35	0.32	1.11	0.15	
# where the first coulumn is the name, second is the formula, third is Cp in metric, fourth is Cv in metric, fifth is Cp in imperial, sixth is Cv in imperial, seventh is gamma, rest we can ignore
# we want to create a dictionary with the following format:
# name, formula, gamma, Cp, Cv, M (molar mass)
# we don't know the molar mass yet


substances = {}


with open("molar_masses.txt","r") as molar_masses:
    for line in molar_masses:
        values = line.split()
        try:
            name = values[0]
            formula = values[1]
            M = float(values[2])
            
            print(name, formula, M)
            substances[name] = {"formula": formula, "M": M}
        except:
            pass


with open("substances.txt","r") as f:
    for line in f:
        values = line.split()
        try:
            name = values[0]
            formula = values[1]
            Cp = float(values[2])
            Cv = float(values[3])
            gamma = Cp/Cv
            if name in substances:
                substances[name]["Cp"] = Cp
                substances[name]["Cv"] = Cv
                substances[name]["gamma"] = gamma
            else:
                substances[name] = {"formula": formula, "gamma": gamma, "Cp": Cp, "Cv": Cv}
        except:
            print(f"Failed to read line: {line}")

# we want to convert the Cp and Cv from kJ/kgK to J/molK

for gas in substances:
    try:
        substances[gas]["Cp"] = substances[gas]["Cp"]*substances[gas]["M"]
        substances[gas]["Cv"] = substances[gas]["Cv"]*substances[gas]["M"]
    except:
        pass



# we want to save the information in a JSON file called "substances.json"
# we want to order the data by name

import json

substances = dict(sorted(substances.items()))

with open("substances.json","w") as f:
    json.dump(substances,f,indent=4)