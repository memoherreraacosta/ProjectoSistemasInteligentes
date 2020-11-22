#------------------------------------------------------------------------------------------------------------------
#   Bayesian network for the burglary alarm problem.
#------------------------------------------------------------------------------------------------------------------

import pyAgrum as gum

# Initialize model
model = gum.BayesNet('CowSickness')

cow1_var = gum.LabelizedVariable('C1', 'C1', 2)
cow1_var.changeLabel(0, 'S')
cow1_var.changeLabel(1, 'H')

cow2_var = gum.LabelizedVariable('C2', 'C2', 2)
cow2_var.changeLabel(0, 'S')
cow2_var.changeLabel(1, 'H')

cow3_var = gum.LabelizedVariable('C3', 'C3', 2)
cow3_var.changeLabel(0, 'S')
cow3_var.changeLabel(1, 'H')

cow4_var = gum.LabelizedVariable('C4', 'C4', 2)
cow4_var.changeLabel(0, 'S')
cow4_var.changeLabel(1, 'H')

cow5_var = gum.LabelizedVariable('C5', 'C5', 2)
cow5_var.changeLabel(0, 'S')
cow5_var.changeLabel(1, 'H')

cow6_var = gum.LabelizedVariable('C6', 'C6', 2)
cow6_var.changeLabel(0, 'S')
cow6_var.changeLabel(1, 'H')

test1_var = gum.LabelizedVariable('T1', 'T1', 2)
test1_var.changeLabel(0, 'F')
test1_var.changeLabel(1, 'T')

test2_var = gum.LabelizedVariable('T2', 'T2', 2)
test2_var.changeLabel(0, 'F')
test2_var.changeLabel(1, 'T')

test3_var = gum.LabelizedVariable('T3', 'T3', 2)
test3_var.changeLabel(0, 'F')
test3_var.changeLabel(1, 'T')

test4_var = gum.LabelizedVariable('T4', 'T4', 2)
test4_var.changeLabel(0, 'F')
test4_var.changeLabel(1, 'T')

test5_var = gum.LabelizedVariable('T5', 'T5', 2)
test5_var.changeLabel(0, 'F')
test5_var.changeLabel(1, 'T')

test6_var = gum.LabelizedVariable('T6', 'T6', 2)
test6_var.changeLabel(0, 'F')
test6_var.changeLabel(1, 'T')



# Build model
cow1 = model.add(cow1_var)
cow2 = model.add(cow2_var)
cow3 = model.add(cow3_var)
cow4 = model.add(cow4_var)
cow5 = model.add(cow5_var)
cow6 = model.add(cow6_var)
test1 = model.add(test1_var)
test2 = model.add(test2_var)
test3 = model.add(test3_var)
test4 = model.add(test4_var)
test5 = model.add(test5_var)
test6 = model.add(test6_var)

model.addArc(cow1, test1)
model.addArc(cow1, cow2)

model.addArc(cow2, test2)
model.addArc(cow2, cow3)

model.addArc(cow3, test3)
model.addArc(cow3, cow4)

model.addArc(cow4, test4)
model.addArc(cow4, cow5)

model.addArc(cow5, test5)
model.addArc(cow5, cow6)

model.addArc(cow6, test6)

# Define conditional probabilities

model.cpt(cow1)[:] = [0.1, 0.9]

model.cpt(test1)[{'C1': 'S'}] = [0.1, 0.9]
model.cpt(test1)[{'C1': 'H'}] = [0.9, 0.1]

model.cpt(cow2)[{'C1':'S'}] = [0.9, 0.1]
model.cpt(cow2)[{'C1':'H'}] = [0.05, 0.95]

model.cpt(test2)[{'C2': 'S'}] = [0.1, 0.9]
model.cpt(test2)[{'C2': 'H'}] = [0.9, 0.1]

model.cpt(cow3)[{'C2':'S'}] = [0.9, 0.1]
model.cpt(cow3)[{'C2':'H'}] = [0.05, 0.95]

model.cpt(test3)[{'C3': 'S'}] = [0.1, 0.9]
model.cpt(test3)[{'C3': 'H'}] = [0.9, 0.1]

model.cpt(cow4)[{'C3':'S'}] = [0.9, 0.1]
model.cpt(cow4)[{'C3':'H'}] = [0.05, 0.95]

model.cpt(test4)[{'C4': 'S'}] = [0.1, 0.9]
model.cpt(test4)[{'C4': 'H'}] = [0.9, 0.1]

model.cpt(cow5)[{'C4':'S'}] = [0.9, 0.1]
model.cpt(cow5)[{'C4':'H'}] = [0.05, 0.95]

model.cpt(test5)[{'C5': 'S'}] = [0.1, 0.9]
model.cpt(test5)[{'C5': 'H'}] = [0.9, 0.1]

model.cpt(cow6)[{'C5':'S'}] = [0.9, 0.1]
model.cpt(cow6)[{'C5':'H'}] = [0.05, 0.95]

model.cpt(test6)[{'C6': 'S'}] = [0.1, 0.9]
model.cpt(test6)[{'C6': 'H'}] = [0.9, 0.1]
ie=gum.LazyPropagation(model)
ie.makeInference()

ie.setEvidence({'T1' : 'F', 'T2' : 'F', 'T3' : 'F', 'T4' : 'F', 'T5' : 'F', 'T6' : 'F'})
ie.addTarget(cow6)
print(ie.posterior(cow6))
