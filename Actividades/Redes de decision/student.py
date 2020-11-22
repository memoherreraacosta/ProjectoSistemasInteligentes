import pyAgrum as gum

# Create a decision network
model = gum.InfluenceDiagram()

# Add a chance node for Lucky
lucky_var = gum.LabelizedVariable('Lucky', 'Lucky', 2)
lucky_var.changeLabel(0, "F")
lucky_var.changeLabel(1, "T")
lucky = model.addChanceNode(lucky_var)

# Add a chance node for Exam
exam_var = gum.LabelizedVariable('Exam', 'Exam', 2)
exam_var.changeLabel(0, "F")
exam_var.changeLabel(1, "T")
exam = model.addChanceNode(exam_var)

# Add a chance node for Lottery
lottery_var = gum.LabelizedVariable('Lottery', 'Lottery', 2)
lottery_var.changeLabel(0, "F")
lottery_var.changeLabel(1, "T")
lottery = model.addChanceNode(lottery_var)

# Add a chance node for Study
study_var = gum.LabelizedVariable('Study', 'Study', 2)
study_var.changeLabel(0, "F")
study_var.changeLabel(1, "T")
study = model.addDecisionNode(study_var)

# Add a decision node for Happiness
happiness_var = gum.LabelizedVariable('Happiness', 'Happiness', 1)
happiness = model.addUtilityNode(happiness_var)

# Add connections between nodes
model.addArc(study, exam)
model.addArc(lucky, exam)
model.addArc(lucky, lottery)
model.addArc(exam, happiness)
model.addArc(lottery, happiness)

# Add conditional probabilities
model.cpt(lucky)[:]=[0.4, 0.6]
model.cpt(exam)[{'Lucky':'F', 'Study': 'F'}] = [0.01, 0.99] 
model.cpt(exam)[{'Lucky':'T', 'Study': 'F'}] = [0.5, 0.5]
model.cpt(exam)[{'Lucky':'T', 'Study': 'T'}] = [0.99, 0.1] 
model.cpt(exam)[{'Lucky':'F', 'Study': 'T'}] = [0.9, 0.1]
model.cpt(lottery)[{'Lucky':'T'}] = [0.6, 0.4]
model.cpt(lottery)[{'Lucky':'F'}] = [0.99, 0.01]

# Add utilities
model.utility(happiness)[{'Lottery':'F', 'Exam':'F'}] = .2
model.utility(happiness)[{'Lottery':'F', 'Exam':'T'}] = .8
model.utility(happiness)[{'Lottery':'T', 'Exam':'F'}] = .6
model.utility(happiness)[{'Lottery':'T', 'Exam':'T'}] = .99

# Create an inference model
ie = gum.InfluenceDiagramInference(model)

# Make an inference with default evidence
ie.makeInference()
print('--- Inference with default evidence ---')
print(ie.displayResult())
print()

# Make an inference when the student doesnt have lucky
ie.setEvidence({'Lucky': 'F'})
ie.makeInference()
print('--- Inference when the student doesnt have lucky ---')
print(ie.displayResult())
print()   

# Make an inference when the student does have lucky
ie.setEvidence({'Lucky': 'T'})
ie.makeInference()
print('--- Inference when the student does have lucky ---')
print(ie.displayResult())
print()