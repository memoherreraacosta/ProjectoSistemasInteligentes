#------------------------------------------------------------------------------------------------------------------
#   Bayesian network for the burglary alarm problem.
#------------------------------------------------------------------------------------------------------------------

import pyAgrum as gum

# Initialize model
model = gum.InfluenceDiagram()

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

sell_var = gum.LabelizedVariable('Sell', 'Sell', 2)
sell_var.changeLabel(0, 'F')
sell_var.changeLabel(1, 'T')


# Build model
cow1 = model.addChanceNode(cow1_var)
cow2 = model.addChanceNode(cow2_var)
cow3 = model.addChanceNode(cow3_var)
cow4 = model.addChanceNode(cow4_var)
cow5 = model.addChanceNode(cow5_var)
cow6 = model.addChanceNode(cow6_var)
test1 = model.addChanceNode(test1_var)
test2 = model.addChanceNode(test2_var)
test3 = model.addChanceNode(test3_var)
test4 = model.addChanceNode(test4_var)
test5 = model.addChanceNode(test5_var)
test6 = model.addChanceNode(test6_var)
sell = model.addDecisionNode(sell_var)

ut_var = gum.LabelizedVariable('Utility','Model Utility',1)
ut = model.addUtilityNode(ut_var)

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
model.addArc(cow6, ut)

model.addArc(sell, ut)

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

model.utility(ut)[{'C6':'S', 'Sell':'F'}] = 0
model.utility(ut)[{'C6':'S', 'Sell':'T'}] = -1000
model.utility(ut)[{'C6':'H', 'Sell':'F'}] = -500
model.utility(ut)[{'C6':'H', 'Sell':'T'}] = 500

ie=gum.InfluenceDiagramInference(model)
ie.makeInference()
print('--- Inference with default evidence ---')
print(ie.displayResult())
print()

ie.setEvidence({'T3': 'T'})
ie.makeInference()
print('--- Inference with T3 = T ---')
print(ie.displayResult())
print()

ie.setEvidence({'T3': 'T', 'T6': 'F'})
ie.makeInference()
print('--- Inference with T3 = T , T6 = F---')
print(ie.displayResult())
print()

ie.setEvidence({'T2': 'F'})
ie.makeInference()
print('--- Inference with T2 = F--')
print(ie.displayResult())
print()

ie.setEvidence({'T2': 'F', 'T4': 'F'})
ie.makeInference()
print('--- Inference with T2 = F, T4 = F--')
print(ie.displayResult())
print()

ie.setEvidence({'T3': 'T', 'T5': 'T', 'T6': 'F'})
ie.makeInference()
print('--- Inference with T3 = T, T5 = T, T6 = F--')
print(ie.displayResult())
print()

ie.setEvidence({'T1': 'T', 'T2': 'T', 'T3': 'T'})
ie.makeInference()
print('--- Inference with T1 = T, T2 = T, T3 = T--')
print(ie.displayResult())
print()

ie.setEvidence({'T1': 'T', 'T2': 'T', 'T3': 'T', 'T6': 'F'})
ie.makeInference()
print('--- Inference with T1 = T, T2 = T, T3 = T, T6 = F--')
print(ie.displayResult())
print()

ie.setEvidence({'T1': 'T', 'T2': 'T', 'T3': 'T', 'T4': 'T', 'T5': 'T', 'T6': 'F'})
ie.makeInference()
print('--- Inference with T1 = T, T2 = T, T3 = T, T4 = T, T5 = T, T6 = F--')
print(ie.displayResult())
print()
